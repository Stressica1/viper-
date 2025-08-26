# GitHub MCP Activation Guide


# GitHub Personal Access Token Setup

## Steps to Create a GitHub PAT:

1. **Go to GitHub Settings**
   - Visit: https://github.com/settings/tokens
   - Click "Generate new token (classic)"

2. **Configure Token Permissions**
   - Note: `classic` tokens are required for MCP
   - Select scopes:
     - [x] `read:packages` - Access GitHub packages
     - [x] `repo` - Full control of private repositories
     - [x] `user` - Read user profile data

3. **Generate and Save Token**
   - Click "Generate token"
   - **IMPORTANT**: Copy the token immediately
   - Token will look like: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

4. **Set Environment Variable**
   ```bash
   export GITHUB_TOKEN=ghp_your_token_here
   ```

5. **Test Token**
   ```bash
   curl -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/user
   ```

## Security Notes:
- Never commit tokens to version control
- Use environment variables, not hardcoded values
- Rotate tokens regularly
- Use the minimum required permissions

**Current Token Status**: Check if GITHUB_TOKEN is set

## Activation Commands

### Direct Docker Run
**Description**: Run MCP server directly with Docker

**Setup**:
```
Set GITHUB_TOKEN environment variable first
```

**Command**:
```
docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN ghcr.io/github/github-mcp-server
```

### Interactive Token Input
**Description**: Run MCP server with token input

**Setup**:
```
Set GITHUB_TOKEN environment variable first
```

**Command**:
```
echo $GITHUB_TOKEN | docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN=/dev/stdin ghcr.io/github/github-mcp-server
```

### VS Code MCP Integration
**Description**: Use VS Code MCP configuration files

**Setup**:
```
Restart VS Code and use MCP commands from palette
```

**Command**:
```
code --enable-proposed-api github.vscode-github-mcp
```

### Background Service Mode
**Description**: Run MCP server in background

**Setup**:
```
Set GITHUB_TOKEN environment variable first
```

**Command**:
```
docker run -d --name github-mcp -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN ghcr.io/github/github-mcp-server
```

