# 🤖 Multi-Agent Deliberation System

A sophisticated multi-agent system for collaborative AI decision-making and analysis using OpenAI, Anthropic, and Google AI models.

## 🎯 Overview

This system orchestrates multiple AI agents with different perspectives and expertise to analyze complex problems, debate solutions, and reach consensus on technical decisions for the Computer Vision Pipeline project.

## 📁 Project Structure

```
multi-agent/
├── configs/
│   └── meeting.yaml              # Agent and meeting configuration
├── agents/
│   ├── prompts/                  # Agent system prompts
│   │   ├── pre_arch_opus.md     # Pre-Architect (Claude Opus)
│   │   ├── v1_prod_team.md      # V1 Production Team
│   │   ├── v2_scientific_team.md # V2 Scientific Team
│   │   ├── cotrr_team.md        # CoTRR Lightweight Team
│   │   ├── tech_analysis_team.md # Technical Analysis
│   │   ├── critic_openai.md     # Critical Evaluator
│   │   ├── integrity_claude.md  # Integrity Guardian
│   │   ├── roundtable_moderator.md # Meeting Moderator
│   │   └── claude_data_analyst.md  # Data Analyst
│   ├── roles.py                  # Agent class definitions
│   └── router.py                 # Message routing logic
├── tools/
│   ├── gemini_search.py         # Gemini search capabilities
│   ├── integrity_rules.py       # Consistency validation
│   ├── parse_actions.py         # Action item extraction
│   ├── collect_artifacts.py     # Report generation
│   └── io_utils.py              # File I/O utilities
├── data/
│   ├── production_evaluation.json
│   └── v2_scientific_review_report.json
├── reports/                      # Auto-generated meeting reports
├── run_meeting.py               # Main orchestrator script
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd multi-agent
pip install -r requirements.txt
pip install pyyaml  # If not already installed
```

### 2. Configure API Keys

Make sure your API keys are set in the parent directory:

```bash
# Already configured in: research/api_keys.env
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
GOOGLE_API_KEY=your-key
```

### 3. Customize Agent Prompts

Edit the prompt files in `agents/prompts/` to define each agent's personality and expertise.

### 4. Run a Meeting

```bash
python run_meeting.py "Should we deploy V1.0 optimizations or pursue V2.0 research?"
```

Or with custom topic:
```bash
python run_meeting.py "What are the top 3 priorities for the pipeline?"
```

## 🎭 Agent Roles

| Agent | Model | Role | Perspective |
|-------|-------|------|-------------|
| **Pre-Architect** | Claude Opus | System Design | High-level architecture |
| **V1 Production Team** | GPT-4 | Production Advocate | Stability & proven results |
| **V2 Scientific Team** | Claude Sonnet | Research Advocate | Innovation & experimentation |
| **CoTRR Team** | Gemini Flash | Lightweight Solutions | Efficiency & practicality |
| **Tech Analysis** | GPT-4 | Technical Evaluation | Deep technical analysis |
| **Critic** | GPT-4 | Devil's Advocate | Challenge assumptions |
| **Integrity Guardian** | Claude Sonnet | Quality Assurance | Consistency & correctness |
| **Data Analyst** | Claude Sonnet | Data-Driven Insights | Metrics & evidence |
| **Moderator** | Claude Opus | Orchestration | Synthesis & facilitation |

## 🔄 Routing Strategies

The system supports multiple routing strategies:

### 1. **Hierarchical** (Recommended)
```
Teams respond → Critic reviews → Integrity checks → Moderator synthesizes
```

### 2. **Broadcast**
All agents receive and respond to the same message simultaneously.

### 3. **Round Robin**
Agents respond in sequence, building on previous responses.

### 4. **Debate**
Teams present positions → Critic challenges each → Synthesis

### 5. **Targeted**
Send messages to specific agents only.

## 🛠️ Key Features

### 1. **Integrity Validation**
- Detects contradictions between agents
- Checks response completeness
- Calculates consensus scores
- Validates decision quality

### 2. **Action Extraction**
Automatically identifies and categorizes:
- Code changes
- Experiments to run
- Analyses to perform
- Decisions to make

### 3. **Artifact Collection**
Auto-generates:
- Meeting transcripts
- Summary reports
- Action item lists
- Integrity reports
- JSON data exports

### 4. **Context Integration**
- Loads project data from `data/` directory
- Provides agents with relevant context
- Maintains conversation history

## 📊 Example Usage

### Basic Meeting

```python
from pathlib import Path
from run_meeting import MeetingOrchestrator

orchestrator = MeetingOrchestrator(Path("configs/meeting.yaml"))

result = orchestrator.run_meeting(
    topic="Should we prioritize V1 optimization or V2 research?",
    strategy="hierarchical",
    rounds=3
)

print(f"Actions: {len(result['actions'])}")
print(f"Consensus: {result['integrity']['metrics']['consensus_score']}")
```

### Quick Query

```python
responses = orchestrator.quick_query(
    "What's the biggest risk in our current approach?"
)

for agent, response in responses.items():
    print(f"{agent}: {response}")
```

## 📈 Output Examples

### Meeting Report Structure

```
reports/
├── transcript_20251012_184500.md    # Full conversation
├── summary_20251012_184500.md       # Executive summary
├── actions_20251012_184500.json     # Extracted actions
├── responses_20251012_184500.json   # Raw responses
└── integrity_20251012_184500.json   # Validation results
```

### Action Items Format

```markdown
# Action Items

## HIGH Priority
- [code_change] Implement caching layer for V1.0 (from tech_analysis)
- [experiment] Validate V2 on clean dataset (from v2_scientific)

## MEDIUM Priority
- [analysis] Compare V1 vs V2 performance metrics (from data_analyst)

## LOW Priority
- [decision] Consider hybrid approach for future (from moderator)
```

## ⚙️ Configuration

Edit `configs/meeting.yaml` to:
- Add/remove agents
- Change models
- Adjust parameters
- Enable/disable tools

```yaml
meeting:
  rounds: 3
  max_tokens_per_response: 2000

agents:
  custom_agent:
    name: "Custom Agent"
    model: "gpt-4"
    provider: "openai"
    role: "Custom role"
    prompt_file: "custom_prompt.md"
```

## 🔧 Advanced Features

### Custom Tools

Add new tools in `tools/` directory:

```python
class CustomTool:
    def process(self, data):
        # Your tool logic
        return result
```

### Custom Routing

Extend `AgentRouter` with new strategies:

```python
def _custom_strategy(self, message):
    # Your routing logic
    return responses
```

## 📝 Best Practices

1. **Prompt Engineering**: Craft clear, specific prompts for each agent
2. **Context Loading**: Provide relevant project data in `data/`
3. **Multiple Rounds**: Use 2-3 rounds for complex decisions
4. **Review Artifacts**: Always check generated reports
5. **Iterate**: Refine prompts based on agent performance

## 🐛 Troubleshooting

### API Errors
- Check API keys are set correctly
- Verify API quotas aren't exceeded
- Check model names are current

### Empty Responses
- Review agent prompts
- Check context is being loaded
- Verify routing strategy is appropriate

### Low Consensus
- This is normal for complex topics!
- Review contradictions in integrity report
- Consider adding more rounds

## 🔜 Future Enhancements

- [ ] Web UI for meeting visualization
- [ ] Real-time streaming responses
- [ ] Agent memory across sessions
- [ ] Custom model fine-tuning
- [ ] Voting/ranking mechanisms
- [ ] Integration with code execution

## 📚 Related Documentation

- [Computer Vision Project](../README.md)
- [V1.0 Production](../docs/04_production_deployment/)
- [Research Timeline](../docs/05_analysis_reports/)

## 🤝 Contributing

To add a new agent:
1. Add configuration to `configs/meeting.yaml`
2. Create prompt file in `agents/prompts/`
3. Update this README

---

**Built with ❤️ using OpenAI, Anthropic, and Google AI**

For questions or issues, refer to the main project documentation.
