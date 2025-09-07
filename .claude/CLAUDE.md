# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Think carefully and implement the most concise solution that changes as little code as possible.

## USE SUB-AGENTS FOR CONTEXT OPTIMIZATION

### 1. Always use the file-analyzer sub-agent when asked to read files.
The file-analyzer agent is an expert in extracting and summarizing critical information from files, particularly log files and verbose outputs. It provides concise, actionable summaries that preserve essential information while dramatically reducing context usage.

### 2. Always use the code-analyzer sub-agent when asked to search code, analyze code, research bugs, or trace logic flow.

The code-analyzer agent is an expert in code analysis, logic tracing, and vulnerability detection. It provides concise, actionable summaries that preserve essential information while dramatically reducing context usage.

### 3. Always use the test-runner sub-agent to run tests and analyze the test results.

Using the test-runner agent ensures:

- Full test output is captured for debugging
- Main conversation stays clean and focused
- Context usage is optimized
- All issues are properly surfaced
- No approval dialogs interrupt the workflow

## Philosophy

### Error Handling

- **Fail fast** for critical configuration (missing text model)
- **Log and continue** for optional features (extraction model)
- **Graceful degradation** when external services unavailable
- **User-friendly messages** through resilience layer

### Testing

- Always use the test-runner agent to execute tests.
- Do not use mock services for anything ever.
- Do not move on to the next test until the current test is complete.
- If the test fails, consider checking if the test is structured correctly before deciding we need to refactor the codebase.
- Tests to be verbose so we can use them for debugging.


## Tone and Behavior

- Criticism is welcome. Please tell me when I am wrong or mistaken, or even when you think I might be wrong or mistaken.
- Please tell me if there is a better approach than the one I am taking.
- Please tell me if there is a relevant standard or convention that I appear to be unaware of.
- Be skeptical.
- Be concise.
- Short summaries are OK, but don't give an extended breakdown unless we are working through the details of a plan.
- Do not flatter, and do not give compliments unless I am specifically asking for your judgement.
- Occasional pleasantries are fine.
- Feel free to ask many questions. If you are in doubt of my intent, don't guess. Ask.

## ABSOLUTE RULES:

- NO PARTIAL IMPLEMENTATION
- NO SIMPLIFICATION : no "//This is simplified stuff for now, complete implementation would blablabla"
- NO CODE DUPLICATION : check existing codebase to reuse functions and constants Read files before writing new functions. Use common sense function name to find them easily.
- NO DEAD CODE : either use or delete from codebase completely
- IMPLEMENT TEST FOR EVERY FUNCTIONS
- NO CHEATER TESTS : test must be accurate, reflect real usage and be designed to reveal flaws. No useless tests! Design tests to be verbose so we can use them for debuging.
- NO INCONSISTENT NAMING - read existing codebase naming patterns.
- NO OVER-ENGINEERING - Don't add unnecessary abstractions, factory patterns, or middleware when simple functions would work. Don't think "enterprise" when you need "working"
- NO MIXED CONCERNS - Don't put validation logic inside API handlers, database queries inside UI components, etc. instead of proper separation
- NO RESOURCE LEAKS - Don't forget to close database connections, clear timeouts, remove event listeners, or clean up file handles

## CODEBASE ARCHITECTURE

This is the **Claude Code Project Management (CCPM) system** - a workflow management tool that transforms PRDs into GitHub issues and coordinates parallel AI agents.

### Core Architecture

```
CCPM Workflow Flow
├── PRD Creation (.claude/prds/) → Brainstorming and requirements
├── Epic Planning (.claude/epics/) → Implementation breakdown
├── GitHub Sync → Issues as single source of truth  
└── Parallel Execution → Multiple agents in Git worktrees
```

### Key Components

1. **Command System** (`.claude/commands/pm/`): Structured workflow commands
   - `/pm:prd-new` - Create comprehensive PRDs through guided brainstorming
   - `/pm:epic-oneshot` - Transform PRD → Epic → GitHub Issues in one command
   - `/pm:issue-start` - Launch specialized agents for implementation
   - `/pm:next` - Intelligent prioritization of next tasks

2. **Agent Coordination**: Context-preserving sub-agents work in parallel
   - Each agent operates in isolation to prevent context pollution
   - Agents coordinate through Git commits and GitHub issue updates
   - Main conversation stays strategic, not bogged down in implementation details

3. **GitHub Integration**: Uses `gh-sub-issue` extension for parent-child relationships
   - Epic issues track sub-task completion automatically
   - Issues serve as the single source of truth for project state
   - Comments provide full audit trail from requirements to code

### Important Directories

- `.claude/commands/pm/` - Project management workflow definitions
- `.claude/agents/` - Specialized context-preserving agents
- `.claude/scripts/pm/` - Shell script implementations of PM commands
- `.claude/epics/` - Local epic workspace (should be in .gitignore)
- `.claude/prds/` - Product requirements documents
- `.claude/context/` - Project-wide context files

### Common Commands

```bash
# Initial setup
/pm:init                    # Install GitHub CLI, authenticate, setup extensions

# Feature development workflow  
/pm:prd-new feature-name    # Create PRD through guided brainstorming
/pm:epic-oneshot feature-name # Transform PRD → Epic → GitHub Issues
/pm:issue-start 1234        # Launch agent to implement specific issue
/pm:next                    # Get next priority task with context

# Status and coordination
/pm:status                  # Overall project dashboard  
/pm:epic-show feature-name  # Display epic progress and tasks
/pm:issue-sync 1234        # Push local progress to GitHub
```

### Testing

- No traditional test framework - this is a workflow management system
- Validation happens through GitHub issue lifecycle and agent coordination
- Use `/pm:validate` to check system integrity
- Monitor agent execution through GitHub issue comments

### Unique Patterns

- **Spec-driven development**: Every code change traces back to documented requirements
- **Parallel agent execution**: Multiple AI agents work simultaneously on different aspects  
- **Context firewalls**: Agents prevent implementation details from polluting main conversation
- **GitHub as database**: No separate PM tools - GitHub Issues are the project state
