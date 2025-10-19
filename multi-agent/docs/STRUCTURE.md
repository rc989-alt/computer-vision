# Multi-Agent System Structure

Complete file listing with descriptions.

## 📂 Directory Structure

```
multi-agent/
├── agents/                          # Agent system
│   ├── prompts/                    # Agent personality prompts
│   │   ├── pre_arch_opus.md       # 🏗️ System Architect (Claude Opus)
│   │   ├── v1_prod_team.md        # 🚀 Production Advocate (GPT-4)
│   │   ├── v2_scientific_team.md  # 🔬 Research Advocate (Claude Sonnet)
│   │   ├── cotrr_team.md          # ⚡ Efficiency Expert (Gemini)
│   │   ├── tech_analysis_team.md  # 🔧 Tech Analyst (GPT-4)
│   │   ├── critic_openai.md       # 🎯 Devil's Advocate (GPT-4)
│   │   ├── integrity_claude.md    # ✅ Quality Guardian (Claude Sonnet)
│   │   ├── roundtable_moderator.md # 🎙️ Moderator (Claude Opus)
│   │   └── claude_data_analyst.md # 📊 Data Analyst (Claude Sonnet)
│   ├── roles.py                    # Agent class definitions
│   └── router.py                   # Message routing & strategies
│
├── configs/
│   └── meeting.yaml                # System configuration
│
├── tools/                          # Utility tools
│   ├── gemini_search.py           # Gemini search capabilities
│   ├── integrity_rules.py         # Consistency validation
│   ├── parse_actions.py           # Action extraction
│   ├── collect_artifacts.py       # Report generation
│   └── io_utils.py                # I/O utilities
│
├── data/                           # Project context data
│   ├── production_evaluation.json  # V1.0 metrics
│   └── v2_scientific_review_report.json # V2.0 findings
│
├── reports/                        # Auto-generated outputs
│   └── (transcripts, summaries, actions, etc.)
│
├── run_meeting.py                  # 🎬 Main orchestrator
├── setup.sh                        # 🛠️ Setup script
├── requirements.txt                # Dependencies
├── README.md                       # Main documentation
└── STRUCTURE.md                    # This file
```

## 🔧 Core Components

### 1. Agent System (`agents/`)

**roles.py** - Core agent implementation
- `AgentConfig`: Configuration dataclass
- `Agent`: Base agent class with API integration
- `AgentTeam`: Team management

**router.py** - Message routing
- `RoutingStrategy`: Enum of strategies
- `Message`: Message dataclass
- `AgentRouter`: Routes messages between agents

### 2. Configuration (`configs/`)

**meeting.yaml** - System settings
- Agent definitions (9 agents)
- Model assignments
- Tool configurations
- Output settings

### 3. Tools (`tools/`)

| Tool | Purpose |
|------|---------|
| `gemini_search.py` | Search & information retrieval |
| `integrity_rules.py` | Validate consistency |
| `parse_actions.py` | Extract action items |
| `collect_artifacts.py` | Generate reports |
| `io_utils.py` | File operations |

### 4. Orchestration

**run_meeting.py** - Main script
- `MeetingOrchestrator`: Coordinates all components
- Runs multi-round deliberations
- Generates comprehensive reports

## 🎯 Agent Roster

| Agent ID | Name | Model | Role |
|----------|------|-------|------|
| `pre_architect` | Pre-Architect | Claude Opus | System design |
| `v1_production` | V1 Production Team | GPT-4 | Production advocacy |
| `v2_scientific` | V2 Scientific Team | Claude Sonnet | Research advocacy |
| `cotrr_team` | CoTRR Team | Gemini Flash | Lightweight solutions |
| `tech_analysis` | Tech Analysis | GPT-4 | Technical evaluation |
| `critic` | Critic | GPT-4 | Critical review |
| `integrity_guardian` | Integrity Guardian | Claude Sonnet | Quality assurance |
| `data_analyst` | Data Analyst | Claude Sonnet | Data analysis |
| `moderator` | Moderator | Claude Opus | Facilitation |

## 🔄 Routing Strategies

1. **Hierarchical** (Default)
   - Teams → Critic → Integrity → Moderator

2. **Broadcast**
   - All agents simultaneously

3. **Round Robin**
   - Sequential with context building

4. **Debate**
   - Proponents → Critic challenges → Synthesis

5. **Targeted**
   - Specific agents only

## 📊 Data Flow

```
1. Load config & context
   ↓
2. Initialize agents
   ↓
3. Route message
   ↓
4. Collect responses
   ↓
5. Validate integrity
   ↓
6. Parse actions
   ↓
7. Generate reports
   ↓
8. Save artifacts
```

## 🚀 Quick Commands

```bash
# Setup
./setup.sh

# Run meeting
python3 run_meeting.py "Your question"

# Custom strategy
python3 run_meeting.py "Question" --strategy=debate

# More rounds
python3 run_meeting.py "Question" --rounds=5
```

## 📝 Customization Points

1. **Agent Prompts** → `agents/prompts/*.md`
2. **Configuration** → `configs/meeting.yaml`
3. **Project Data** → `data/*.json`
4. **Tools** → `tools/*.py`
5. **Routing Logic** → `agents/router.py`

## 🎨 Output Artifacts

Each meeting generates:
- `transcript_*.md` - Full conversation
- `summary_*.md` - Executive summary
- `actions_*.json` - Action items
- `responses_*.json` - Raw responses
- `integrity_*.json` - Validation results

## 🔮 Extension Ideas

- Add more agents with specialized roles
- Implement voting mechanisms
- Add streaming responses
- Build web UI
- Integration with code execution
- Memory across sessions
- Custom tool integration

## 📚 File Count

- **Python files**: 8
- **Prompt files**: 9
- **Config files**: 1
- **Data files**: 2
- **Documentation**: 3
- **Total**: 23 files

---

**Status**: ✅ Complete and ready to use!

Next step: Add your custom prompts and run your first meeting.
