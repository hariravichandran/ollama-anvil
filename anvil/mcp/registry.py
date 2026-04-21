"""Registry of known MCP servers — built-in and community."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MCPEntry",
    "MCP_REGISTRY",
    "VALID_CATEGORIES",
    "search_registry",
    "suggest_mcps",
    "MAX_SEARCH_QUERY_LENGTH",
]


@dataclass(slots=True)
class MCPEntry:
    """An MCP server definition."""

    name: str
    description: str
    category: str  # "search", "data", "tools", "productivity", "development", "cloud", "communication", "ai"
    builtin: bool  # ships with ollama-anvil
    install_cmd: str  # how to install (pip, npm, etc.)
    package: str  # package name
    config_example: dict  # example configuration


# Built-in and well-known MCP servers
MCP_REGISTRY: dict[str, MCPEntry] = {
    # ─── Search & Web ────────────────────────────────────────────────────────
    "web-search": MCPEntry(
        name="web-search",
        description="Search the web using DuckDuckGo (no API key required)",
        category="search",
        builtin=True,
        install_cmd="",
        package="",
        config_example={"enabled": True, "max_results": 5, "cache_ttl": 21600},
    ),
    "brave-search": MCPEntry(
        name="brave-search",
        description="Web search using the Brave Search API",
        category="search",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-brave-search",
        package="@modelcontextprotocol/server-brave-search",
        config_example={"api_key_env": "BRAVE_API_KEY"},
    ),
    "fetch": MCPEntry(
        name="fetch",
        description="Fetch and extract content from web pages",
        category="search",
        builtin=False,
        install_cmd="pip install mcp-server-fetch",
        package="mcp-server-fetch",
        config_example={},
    ),
    "tavily": MCPEntry(
        name="tavily",
        description="AI-powered search engine for accurate, real-time results",
        category="search",
        builtin=False,
        install_cmd="npm install -g tavily-mcp",
        package="tavily-mcp",
        config_example={"api_key_env": "TAVILY_API_KEY"},
    ),
    "exa": MCPEntry(
        name="exa",
        description="Neural search engine — find content by meaning, not just keywords",
        category="search",
        builtin=False,
        install_cmd="npm install -g exa-mcp-server",
        package="exa-mcp-server",
        config_example={"api_key_env": "EXA_API_KEY"},
    ),

    # ─── File & Data ─────────────────────────────────────────────────────────
    "filesystem": MCPEntry(
        name="filesystem",
        description="Read, write, and manage files on disk",
        category="tools",
        builtin=True,
        install_cmd="",
        package="",
        config_example={"enabled": True, "root_dir": "."},
    ),
    "memory": MCPEntry(
        name="memory",
        description="Persistent knowledge graph for long-term memory across sessions",
        category="tools",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-memory",
        package="@modelcontextprotocol/server-memory",
        config_example={"storage_path": "~/.config/ollama-anvil/memory"},
    ),
    "pandoc": MCPEntry(
        name="pandoc",
        description="Document conversion — convert between Markdown, PDF, DOCX, HTML, and more",
        category="tools",
        builtin=False,
        install_cmd="npm install -g mcp-pandoc",
        package="mcp-pandoc",
        config_example={},
    ),
    "everart": MCPEntry(
        name="everart",
        description="AI image generation and manipulation",
        category="tools",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-everart",
        package="@modelcontextprotocol/server-everart",
        config_example={"api_key_env": "EVERART_API_KEY"},
    ),

    # ─── Databases ───────────────────────────────────────────────────────────
    "sqlite": MCPEntry(
        name="sqlite",
        description="Query and manage SQLite databases",
        category="data",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-sqlite",
        package="@modelcontextprotocol/server-sqlite",
        config_example={"db_path": "example.db"},
    ),
    "postgres": MCPEntry(
        name="postgres",
        description="Query and manage PostgreSQL databases",
        category="data",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-postgres",
        package="@modelcontextprotocol/server-postgres",
        config_example={"connection_string_env": "DATABASE_URL"},
    ),
    "mysql": MCPEntry(
        name="mysql",
        description="Query and manage MySQL databases",
        category="data",
        builtin=False,
        install_cmd="npm install -g @benborla29/mcp-server-mysql",
        package="@benborla29/mcp-server-mysql",
        config_example={"connection_string_env": "MYSQL_URL"},
    ),
    "redis": MCPEntry(
        name="redis",
        description="Redis key-value store operations",
        category="data",
        builtin=False,
        install_cmd="npm install -g @punkpeye/mcp-server-redis",
        package="@punkpeye/mcp-server-redis",
        config_example={"url": "redis://localhost:6379"},
    ),
    "mongodb": MCPEntry(
        name="mongodb",
        description="Query and manage MongoDB databases",
        category="data",
        builtin=False,
        install_cmd="npm install -g mcp-mongo-server",
        package="mcp-mongo-server",
        config_example={"connection_string_env": "MONGODB_URI"},
    ),

    # ─── Development Tools ───────────────────────────────────────────────────
    "github": MCPEntry(
        name="github",
        description="Interact with GitHub repos, issues, PRs, and actions",
        category="development",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-github",
        package="@modelcontextprotocol/server-github",
        config_example={"token_env": "GITHUB_TOKEN"},
    ),
    "gitlab": MCPEntry(
        name="gitlab",
        description="Interact with GitLab repos, issues, merge requests, and CI/CD",
        category="development",
        builtin=False,
        install_cmd="npm install -g @zereight/mcp-gitlab",
        package="@zereight/mcp-gitlab",
        config_example={"token_env": "GITLAB_TOKEN", "base_url": "https://gitlab.com"},
    ),
    "docker": MCPEntry(
        name="docker",
        description="Manage Docker containers, images, and compose stacks",
        category="development",
        builtin=False,
        install_cmd="npm install -g mcp-docker",
        package="mcp-docker",
        config_example={},
    ),
    "sentry": MCPEntry(
        name="sentry",
        description="Access Sentry error tracking — view issues, stack traces, and events",
        category="development",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-sentry",
        package="@modelcontextprotocol/server-sentry",
        config_example={"token_env": "SENTRY_AUTH_TOKEN"},
    ),

    # ─── Productivity ────────────────────────────────────────────────────────
    "slack": MCPEntry(
        name="slack",
        description="Send and read Slack messages, manage channels",
        category="productivity",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-slack",
        package="@modelcontextprotocol/server-slack",
        config_example={"token_env": "SLACK_BOT_TOKEN"},
    ),
    "google-drive": MCPEntry(
        name="google-drive",
        description="Search, read, and manage Google Drive files",
        category="productivity",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-google-drive",
        package="@modelcontextprotocol/server-google-drive",
        config_example={"credentials_env": "GOOGLE_CREDENTIALS"},
    ),
    "notion": MCPEntry(
        name="notion",
        description="Read and manage Notion pages, databases, and workspaces",
        category="productivity",
        builtin=False,
        install_cmd="npm install -g notion-mcp-server",
        package="notion-mcp-server",
        config_example={"token_env": "NOTION_TOKEN"},
    ),
    "linear": MCPEntry(
        name="linear",
        description="Manage Linear issues, projects, and workflows",
        category="productivity",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-linear",
        package="@modelcontextprotocol/server-linear",
        config_example={"api_key_env": "LINEAR_API_KEY"},
    ),
    "todoist": MCPEntry(
        name="todoist",
        description="Manage Todoist tasks, projects, and labels",
        category="productivity",
        builtin=False,
        install_cmd="npm install -g todoist-mcp-server",
        package="todoist-mcp-server",
        config_example={"api_key_env": "TODOIST_API_KEY"},
    ),

    # ─── Browser Automation ──────────────────────────────────────────────────
    "puppeteer": MCPEntry(
        name="puppeteer",
        description="Browser automation — navigate pages, take screenshots, fill forms",
        category="tools",
        builtin=False,
        install_cmd="npm install -g @modelcontextprotocol/server-puppeteer",
        package="@modelcontextprotocol/server-puppeteer",
        config_example={},
    ),
    "playwright": MCPEntry(
        name="playwright",
        description="Advanced browser automation with multi-browser support",
        category="tools",
        builtin=False,
        install_cmd="npm install -g @anthropic/mcp-playwright",
        package="@anthropic/mcp-playwright",
        config_example={"browser": "chromium"},
    ),

    # ─── Cloud & Infrastructure ──────────────────────────────────────────────
    "aws": MCPEntry(
        name="aws",
        description="Manage AWS services — S3, EC2, Lambda, CloudFormation, and more",
        category="cloud",
        builtin=False,
        install_cmd="npm install -g aws-mcp",
        package="aws-mcp",
        config_example={"profile": "default", "region": "us-east-1"},
    ),
    "cloudflare": MCPEntry(
        name="cloudflare",
        description="Manage Cloudflare Workers, DNS, and R2 storage",
        category="cloud",
        builtin=False,
        install_cmd="npm install -g @cloudflare/mcp-server-cloudflare",
        package="@cloudflare/mcp-server-cloudflare",
        config_example={"token_env": "CLOUDFLARE_API_TOKEN"},
    ),
    "vercel": MCPEntry(
        name="vercel",
        description="Manage Vercel deployments, projects, and domains",
        category="cloud",
        builtin=False,
        install_cmd="npm install -g vercel-mcp",
        package="vercel-mcp",
        config_example={"token_env": "VERCEL_TOKEN"},
    ),

    # ─── Communication ───────────────────────────────────────────────────────
    "discord": MCPEntry(
        name="discord",
        description="Read and send Discord messages, manage channels",
        category="communication",
        builtin=False,
        install_cmd="npm install -g discord-mcp-server",
        package="discord-mcp-server",
        config_example={"token_env": "DISCORD_BOT_TOKEN"},
    ),
}


# Valid MCP categories
VALID_CATEGORIES = {"search", "data", "tools", "productivity", "development", "cloud", "communication", "ai"}

# Query limits
MAX_SEARCH_QUERY_LENGTH = 200


def search_registry(query: str) -> list[MCPEntry]:
    """Search the MCP registry by name, description, or category."""
    if not query or not query.strip():
        return list(MCP_REGISTRY.values())
    # Truncate long queries
    query = query[:MAX_SEARCH_QUERY_LENGTH]
    query_lower = query.lower()
    seen = set()  # dedup by name
    results = []
    for entry in MCP_REGISTRY.values():
        if entry.name in seen:
            continue
        if (query_lower in entry.name.lower()
                or query_lower in entry.description.lower()
                or query_lower in entry.category.lower()):
            results.append(entry)
            seen.add(entry.name)
    return results


def suggest_mcps(context: str) -> list[MCPEntry]:
    """Suggest MCPs based on conversation context.

    Simple keyword matching — can be enhanced with LLM-based suggestions.
    """
    if not context or not context.strip():
        return []
    # Truncate long context to avoid slow processing
    context_lower = context[:5000].lower()

    suggestions = []
    seen = set()  # dedup
    keyword_map = {
        "github": ["github", "repo", "pull request", "issue", "PR"],
        "gitlab": ["gitlab", "merge request", "mr", "ci/cd"],
        "sqlite": ["sqlite", "database", "sql", "query", "db file"],
        "postgres": ["postgres", "postgresql", "database"],
        "mysql": ["mysql", "mariadb"],
        "mongodb": ["mongodb", "mongo", "nosql", "document database"],
        "redis": ["redis", "cache", "key-value"],
        "puppeteer": ["browser", "screenshot", "web page", "scrape", "headless"],
        "playwright": ["browser automation", "end-to-end test", "e2e"],
        "brave-search": ["search", "find online", "look up", "brave"],
        "tavily": ["research", "accurate search", "ai search"],
        "slack": ["slack", "message", "channel"],
        "discord": ["discord", "server", "bot"],
        "memory": ["remember", "recall", "knowledge", "long-term"],
        "fetch": ["fetch", "download", "web page", "url", "http"],
        "docker": ["docker", "container", "compose", "image"],
        "google-drive": ["google drive", "gdrive", "google docs"],
        "notion": ["notion", "wiki", "notes"],
        "linear": ["linear", "issue tracker", "project management"],
        "todoist": ["todoist", "todo", "task list"],
        "sentry": ["sentry", "error tracking", "crash", "stack trace"],
        "aws": ["aws", "amazon", "s3", "ec2", "lambda"],
        "cloudflare": ["cloudflare", "workers", "dns", "cdn"],
        "pandoc": ["convert", "pdf", "docx", "document conversion"],
        "vercel": ["vercel", "deploy", "deployment", "serverless"],
        "exa": ["exa", "neural search", "semantic search"],
        "everart": ["everart", "image generation", "generate image", "ai art"],
        "filesystem": ["file system", "read file", "write file", "file operations"],
        "web-search": ["web search", "search online", "duckduckgo"],
    }

    for mcp_name, keywords in keyword_map.items():
        if mcp_name in seen:
            continue
        if any(kw in context_lower for kw in keywords):
            entry = MCP_REGISTRY.get(mcp_name)
            if entry:
                suggestions.append(entry)
                seen.add(mcp_name)

    return suggestions
