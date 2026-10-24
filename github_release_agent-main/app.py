"""
GitHub Release Documentation Manager (Streamlit Edition)
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
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="GitHub Changelog MCP Agent",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 GitHub Release Documentation Manager")
st.caption("Created by Swapna V — Automated MCP Agent for Changelog & PR Generation")

# Safely load defaults from environment or Streamlit secrets (if available)
default_pat = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
default_gemini_key = os.getenv("GOOGLE_API_KEY", "")

try:
    if "GITHUB_PERSONAL_ACCESS_TOKEN" in st.secrets:
        default_pat = st.secrets["GITHUB_PERSONAL_ACCESS_TOKEN"]
    if "GOOGLE_API_KEY" in st.secrets:
        default_gemini_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    # Safely bypass if secrets.toml does not exist locally
    pass

# Sidebar for Credentials
with st.sidebar:
    st.header("🔑 Configuration")
    
    github_pat = st.text_input("GitHub Personal Access Token (PAT)", value=default_pat, type="password")
    google_api_key = st.text_input("Google Gemini API Key", value=default_gemini_key, type="password")
    
    st.markdown("---")
    st.markdown("### 📌 Instructions")
    st.markdown("1. Provide your GitHub PAT and Google API Key.")
    st.markdown("2. Enter the target GitHub Repository URL.")
    st.markdown("3. Click **Generate Changelog & PR** to run the MCP workflow.")

# Main Input Form
with st.form("changelog_form"):
    repo_url = st.text_input(
        "GitHub Repository URL", 
        placeholder="https://github.com/owner/repo"
    )
    col1, col2 = st.columns(2)
    with col1:
        version_tag = st.text_input("Version Tag", value="1.1.0")
    with col2:
        default_branch = st.text_input("Default Branch", value="main")
    
    submitted = st.form_submit_button("✨ Generate Changelog & PR", use_container_width=True)


def parse_repo_url(url: str):
    """Extracts owner and repo name from GitHub URL."""
    pattern = r"github\.com/([^/]+)/([^/.]+)"
    match = re.search(pattern, str(url))
    if not match:
        raise ValueError("Invalid GitHub Repository URL. Format: https://github.com/owner/repo")
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


async def run_changelog_agent(pat, r_url, v_tag, d_branch, g_key):
    os.environ["GOOGLE_API_KEY"] = g_key
    owner, repo = parse_repo_url(r_url)
    new_branch = f"docs/changelog-v{v_tag}"

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={
            **os.environ,
            "GITHUB_PERSONAL_ACCESS_TOKEN": pat,
        },
    )

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
            new_entry = f"\n## [{v_tag}] - {today_str}\n"

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
            target_branch = d_branch
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
                target_branch = "master" if d_branch == "main" else "main"
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
                    "message": f"docs: update CHANGELOG.md for v{v_tag}",
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
                    "title": f"docs: update CHANGELOG.md for v{v_tag} release",
                    "body": "Automated weekly changelog update generated by GitHub Release Documentation Manager (Swapna V).",
                    "head": new_branch,
                    "base": target_branch,
                },
            )

            pr_url = pr_result.get("html_url") if isinstance(pr_result, dict) else None

            return {
                "status": "success",
                "pr_url": pr_url,
                "version": v_tag,
                "summary": new_entry,
            }


# Trigger Execution on Button Click
if submitted:
    if not github_pat or not google_api_key or not repo_url:
        st.error("⚠️ Please fill in all required fields (GitHub PAT, Google API Key, and Repository URL).")
    else:
        with st.spinner("🔄 Running MCP agent, analyzing commits/PRs, updating CHANGELOG, and opening PR..."):
            try:
                result = asyncio.run(
                    run_changelog_agent(
                        github_pat, str(repo_url), version_tag, default_branch, google_api_key
                    )
                )

                if result.get("status") == "warning":
                    st.warning(result.get("message"))
                else:
                    st.success("🎉 Changelog generated and Pull Request created successfully!")
                    if result.get("pr_url"):
                        st.markdown(f"### [🔗 View Pull Request on GitHub]({result.get('pr_url')})")
                    
                    st.markdown("### 📝 Generated Changelog Entry")
                    st.code(result.get("summary"), language="markdown")

            except Exception as e:
                st.error("❌ An error occurred during MCP execution:")
                st.exception(e)
