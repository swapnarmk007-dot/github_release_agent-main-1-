"""
GitHub Release Documentation Manager
Created by Swapna V
"""

import asyncio
import base64
import json
import os
import re
import traceback
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, HttpUrl
from fastapi.responses import FileResponse

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is missing from .env file.")

app = FastAPI(title="GitHub Changelog MCP Agent API - Created by Swapna V")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChangelogRequest(BaseModel):
    github_pat: str
    repo_url: HttpUrl
    version_tag: str = "1.1.0"
    default_branch: str = "main"


def parse_repo_url(url: str):
    """Extracts owner and repo name from GitHub URL."""
    pattern = r"github\.com/([^/]+)/([^/.]+)"
    match = re.search(pattern, str(url))
    if not match:
        raise HTTPException(
            status_code=400,
            detail="Invalid GitHub Repository URL. Format: https://github.com/owner/repo",
        )
    return match.group(1), match.group(2).replace(".git", "")


def extract_file_content(file_info) -> str:
    """Safely decodes Base64 or plain-text file content from GitHub MCP response."""
    if not isinstance(file_info, dict):
        return ""
    content = file_info.get("content", "")
    encoding = file_info.get("encoding", "")

    if encoding == "base64":
        try:
            return base64.b64decode(content).decode("utf-8")
        except Exception:
            return ""
    return content


def categorize_item_with_rules(title: str, labels: list) -> str:
    title_lower = title.lower()
    labels_lower = [l.lower() for l in labels]

    if any(l in labels_lower for l in ["feature", "enhancement", "feat"]):
        return "Added"
    if any(l in labels_lower for l in ["bug", "fix", "bugfix"]):
        return "Fixed"
    if any(l in labels_lower for l in ["security", "vulnerability"]):
        return "Security"
    if any(l in labels_lower for l in ["refactor", "chore", "style"]):
        return "Changed"

    if title_lower.startswith(("feat:", "feature:")):
        return "Added"
    if title_lower.startswith(("fix:", "bugfix:")):
        return "Fixed"
    if title_lower.startswith("sec:"):
        return "Security"
    if title_lower.startswith(("refactor:", "style:", "chore:")):
        return "Changed"

    return "LLM_FALLBACK"


async def call_mcp_tool(session: ClientSession, tool_name: str, arguments: dict):
    """Helper to call MCP tools and return parsed JSON or raw responses."""
    result = await session.call_tool(tool_name, arguments)
    if result.content and len(result.content) > 0:
        raw_text = result.content[0].text
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return raw_text
    return None


@app.get("/", response_class=FileResponse)
async def read_index():
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse("index.html")


@app.post("/generate-changelog")
async def generate_changelog(payload: ChangelogRequest):
    owner, repo = parse_repo_url(str(payload.repo_url))
    new_branch = f"docs/changelog-v{payload.version_tag}"

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={
            **os.environ,
            "GITHUB_PERSONAL_ACCESS_TOKEN": payload.github_pat,
        },
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 1. Fetch closed PRs
                prs_data = await call_mcp_tool(
                    session,
                    "list_pull_requests",
                    {"owner": owner, "repo": repo, "state": "closed"},
                )

                seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
                changelog_items = []

                if isinstance(prs_data, list):
                    for pr in prs_data:
                        merged_at_str = pr.get("merged_at")
                        if merged_at_str:
                            merged_at = datetime.fromisoformat(
                                merged_at_str.replace("Z", "+00:00")
                            )
                            if merged_at >= seven_days_ago:
                                labels = [
                                    l.get("name", "")
                                    for l in pr.get("labels", [])
                                ]
                                changelog_items.append(
                                    {
                                        "title": pr["title"],
                                        "body": pr.get("body", ""),
                                        "labels": labels,
                                        "ref": f"#{pr['number']}",
                                    }
                                )

                # Fallback: Fetch commits if no merged PRs exist
                if not changelog_items:
                    commits_data = await call_mcp_tool(
                        session,
                        "list_commits",
                        {
                            "owner": owner,
                            "repo": repo,
                            "since": seven_days_ago.isoformat(),
                        },
                    )
                    if isinstance(commits_data, list) and commits_data:
                        for commit in commits_data:
                            msg = (
                                commit.get("commit", {})
                                .get("message", "")
                                .split("\n")[0]
                            )
                            sha_short = commit.get("sha", "")[:7]
                            changelog_items.append(
                                {
                                    "title": msg,
                                    "body": "",
                                    "labels": [],
                                    "ref": f"({sha_short})",
                                }
                            )

                if not changelog_items:
                    return {
                        "status": "warning",
                        "message": "No merged PRs or direct commits found in the last 7 days.",
                    }

                # 2. Categorize items
                categorized_entries = {
                    "Added": [],
                    "Fixed": [],
                    "Changed": [],
                    "Security": [],
                }

                for item in changelog_items:
                    category = categorize_item_with_rules(
                        item["title"], item["labels"]
                    )

                    if category == "LLM_FALLBACK":
                        prompt = (
                            "Classify this change entry into ONE category: [Added, Fixed, Changed, Security].\n"
                            f"Title: {item['title']}\nBody: {item['body']}\n"
                            "Respond with ONLY the category name."
                        )
                        response = await llm.ainvoke([HumanMessage(content=prompt)])

                        if isinstance(response.content, list):
                            raw_cat = "".join(
                                [
                                    block if isinstance(block, str) else block.get("text", "")
                                    for block in response.content
                                ]
                            )
                        else:
                            raw_cat = str(response.content)

                        category = raw_cat.strip()

                    if category not in categorized_entries:
                        categorized_entries[category] = []

                    categorized_entries[category].append(
                        f"- {item['title']} {item['ref']}"
                    )

                # 3. Read existing CHANGELOG.md
                existing_changelog = ""
                try:
                    file_info = await call_mcp_tool(
                        session,
                        "get_file_contents",
                        {"owner": owner, "repo": repo, "path": "CHANGELOG.md"},
                    )
                    existing_changelog = extract_file_content(file_info)
                except Exception:
                    existing_changelog = ""

                if not existing_changelog:
                    existing_changelog = "# Changelog\n\nAll notable changes are documented here.\n"

                # 4. Synthesize new version entry
                today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                new_entry = f"\n## [{payload.version_tag}] - {today_str}\n"

                for cat, entries in categorized_entries.items():
                    if entries:
                        new_entry += f"\n### {cat}\n"
                        for e in entries:
                            new_entry += f"{e}\n"

                if "# Changelog" in existing_changelog:
                    updated_changelog = existing_changelog.replace(
                        "# Changelog\n", f"# Changelog\n{new_entry}"
                    )
                else:
                    updated_changelog = (
                        f"# Changelog\n{new_entry}\n\n" + existing_changelog
                    )

                # 5. Create Branch
                target_branch = payload.default_branch
                try:
                    await call_mcp_tool(
                        session,
                        "create_branch",
                        {
                            "owner": owner,
                            "repo": repo,
                            "branch": new_branch,
                            "from_branch": target_branch,
                        },
                    )
                except Exception:
                    target_branch = "master" if payload.default_branch == "main" else "main"
                    await call_mcp_tool(
                        session,
                        "create_branch",
                        {
                            "owner": owner,
                            "repo": repo,
                            "branch": new_branch,
                            "from_branch": target_branch,
                        },
                    )

                # 6. Commit CHANGELOG.md to new branch
                await call_mcp_tool(
                    session,
                    "create_or_update_file",
                    {
                        "owner": owner,
                        "repo": repo,
                        "path": "CHANGELOG.md",
                        "message": f"docs: update CHANGELOG.md for v{payload.version_tag}",
                        "content": updated_changelog,
                        "branch": new_branch,
                    },
                )

                # 7. Create Pull Request
                pr_result = await call_mcp_tool(
                    session,
                    "create_pull_request",
                    {
                        "owner": owner,
                        "repo": repo,
                        "title": f"docs: update CHANGELOG.md for v{payload.version_tag} release",
                        "body": "Automated weekly changelog update generated by GitHub Release Documentation Manager (Swapna V).",
                        "head": new_branch,
                        "base": target_branch,
                    },
                )

                pr_url = pr_result.get("html_url") if isinstance(pr_result, dict) else None

                return {
                    "status": "success",
                    "pr_url": pr_url,
                    "version": payload.version_tag,
                    "summary": new_entry,
                }

    except Exception as e:
        print("\n=================== DETAILED AGENT ERROR ===================")
        traceback.print_exc()
        print("============================================================\n")
        raise HTTPException(
            status_code=500, detail=f"GitHub MCP Execution Error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=False)