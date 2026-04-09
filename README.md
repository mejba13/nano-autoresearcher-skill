# nano-autoresearcher-skill

A Claude skill for autonomous ML research in [Karpathy's autoresearch repo](https://github.com/karpathy/autoresearch) — iterative 5-minute GPU experiments to minimize `val_bpb`.

## What it does

This skill teaches Claude to operate as an autonomous ML research agent. Drop Claude into the autoresearch repo on any machine, say "start a new autoresearch session," and Claude will:

1. Run preflight checks (GPU, data, dependencies)
2. Establish a baseline `val_bpb` on unmodified `train.py`
3. Propose and execute experiments — editing hyperparameters, architecture, optimizer settings
4. Keep improvements, revert failures, log everything to `results.tsv`
5. Loop indefinitely (~12 experiments/hour, ~100 overnight)

## Install

### Claude Code

```bash
claude install-skill /path/to/nano-autoresearcher-skill
```

Or copy the `SKILL.md` into your `.claude/skills/nano-autoresearcher/` directory.

### Cowork

Install the `.skill` file directly from the repo:

```
nano-autoresearcher.skill
```

## What's in this repo

```
├── SKILL.md                    # The skill itself (main file)
├── nano-autoresearcher.skill   # Packaged .skill file for easy install
├── evals/
│   └── evals.json              # Test prompts for evaluating the skill
├── benchmark/
│   ├── benchmark.json          # Quantitative eval results (100% pass rate)
│   └── eval-viewer.html        # Interactive side-by-side comparison viewer
└── README.md
```

## Benchmark results

Tested across 3 scenarios with 22 assertions total:

| Eval | With Skill | Without Skill |
|------|-----------|--------------|
| Session setup plan | 9/9 (100%) | 3/9 (33%) |
| Next experiment ideas | 6/6 (100%) | 4/6 (67%) |
| Small hardware mode | 7/7 (100%) | 1/7 (14%) |
| **Overall** | **22/22 (100%)** | **8/22 (36%)** |

The skill is most impactful for setup (correct toolchain, git workflow, results format) and hardware-specific configuration (exact hyperparameter values for constrained GPUs).

## Key features

- **Grounded in actual code** — every parameter name, default value, and line number matches the real `train.py` and `prepare.py`
- **Exploration strategy** — teaches Claude a proven 4-phase sequence (optimizer LRs → model scale → schedule tuning → architecture)
- **Complete knob reference** — hyperparameter constants + in-code values (softcap, MLP activation, Muon momentum ramp, etc.)
- **Small-hardware mode** — specific values for RTX 3060/4060 class GPUs, with TinyStories fallback
- **Works in both Claude Code and Cowork**

## Requirements

- [Karpathy's autoresearch repo](https://github.com/karpathy/autoresearch)
- NVIDIA GPU with CUDA + Flash Attention 3 support
- `uv` package manager
- Claude Code or Cowork with this skill installed

## License

MIT

---

## Developed By

<div align="center">

<img width="380" height="420" alt="engr-mejba-ahmed" src="https://github.com/user-attachments/assets/83e72c39-5eaa-428a-884b-cb4714332487" />


### **Engr Mejba Ahmed**

**AI Developer | Software Engineer | Entrepreneur**

[![Portfolio](https://img.shields.io/badge/Portfolio-mejba.me-10B981?style=for-the-badge&logo=google-chrome&logoColor=white)](https://www.mejba.me)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/mejba)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mejba13)

</div>

---

## Hire / Work With Me

I build AI-powered applications, mobile apps, and enterprise solutions. Let's bring your ideas to life!

| Platform | Description | Link |
|----------|-------------|------|
| **Fiverr** | Custom builds, integrations, performance optimization | [fiverr.com/s/EgxYmWD](https://www.fiverr.com/s/EgxYmWD) |
| **Mejba Personal Portfolio** | Full portfolio & contact | [mejba.me](https://www.mejba.me) |
| **Ramlit Limited** | Software development company | [ramlit.com](https://www.ramlit.com) |
| **ColorPark Creative Agency** | UI/UX & creative solutions | [colorpark.io](https://www.colorpark.io) |
| **xCyberSecurity** | Global cybersecurity services | [xcybersecurity.io](https://www.xcybersecurity.io) |

---

Prepared by:

**Engr Mejba Ahmed**
Software Engineer | AI Developer | Cloud DevOps Engineer
[https://mejba.me](https://mejba.me)
