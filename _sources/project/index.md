# 🎓 Final Project

**Deadline: Friday 19 June 2026, 17:00**

The final project is **group work** (groups of 4 students). Groups self-organise — aim for a mix of capabilities. The same groups are used for both the debate and the final report.

The project aims to develop a **global climate mitigation strategy** from the perspective of your self-assigned actor. Each group adopts a specific role and designs a strategy in line with their assigned actor, but in light of the multi-actor complexity of the case. As part of the project, there are two structured debates (see [Debate](../debate/index.md)).

---

## Report structure

The final report has **two parts**:

| Part | Title | Weight |
|------|-------|--------|
| 1 | Model-based analysis and policy advice | **70 %** |
| 2 | Political reflection on the COP31 debate | **30 %** |

**Maximum length:** 10,000 words for the main text (excluding references and appendices). Figures and tables count as 300 words each. Being over the word limit can result in a lower grade.

**Style:** Written for a technically-minded audience. Bottom line up front (BLUF). Aim for 10–20 pages.

---

## Part 1 — Model-based analysis and policy advice (70 %)

### Goal

Show that you can structure a messy decision problem, analyse it using carefully selected DMDU techniques, and render advice to policy makers.

### Suggested outline

**Executive summary** *(1 page)*
Advice for the problem owner — understandable for a general audience unfamiliar with deep uncertainty methods. What advice do you give, and why?

**Problem framing**
The decision problem can be structured in many ways. How are you framing it? What do you see as the key objectives and constraints? What levers are relevant? What is treated as uncertain? Show awareness of the political arena your problem owner operates in. You may entertain more than one problem formulation.

**Approach**
What selection of deep uncertainty methods are you using, in what order, and why? Clearly motivate and ground in the literature.

**Results**
A readable summary of results. Don't pursue death by figures — carefully select visualisations that tell your story and logically lead to the conclusions and policy advice.

**Discussion**
What are the key threats to the validity of your conclusions? What directions do you see for further refinement? Ground in literature and awareness of the decision arena.

**Conclusions**
Extended conclusion grounded in your results and discussion, leading to clear advice for your problem owner.

### Code and analyses

The code and analyses are an integral part of the assignment. The **primary focus of grading is on reproducibility**.

Checklist:
- Is there a README explaining dependencies and repository structure?
- Are the datasets required to reproduce the advice included?
- Can all figures and tables in the report be traced to specific notebooks?
- Is the code annotated and are results interpreted?
- Is the analysis process tractable from the notebooks/scripts?
- Does the code follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and the [NumPy doc standard](https://numpydoc.readthedocs.io/en/latest/format.html)?

Use **literate computing** in Jupyter notebooks: interleave code, outputs, text, and LaTeX equations so someone else can follow your thinking.

Submit code as either a GitHub repository link or a zip file shared via a file transfer service.

### Rubric — Part 1

| Criterion | Insufficient | Weak | Sufficient | Good | Excellent |
|-----------|-------------|------|------------|------|-----------|
| **Executive summary** | No advice | Unclear or inaccessible advice | Clear advice understandable to non-expert | Clear, grounded in multi-actor context | Convincing advice in multi-actor context, understandable for non-experts |
| **Problem framing** | No clear framing | Naïve framing, no awareness of policy arena | Sensible framing in light of problem owner | Attention to multiple key actors | Explicit rival framings from multiple perspectives |
| **Approach** | Single technique, no literature | Limited techniques, poorly grounded | Standard DMDU approach (e.g. RDM), grounded in literature | Multiple framings or state-of-the-art techniques, motivated | Analysis across rival framings using state-of-the-art techniques |
| **Results** | Death by graphs, no narrative | Too many graphs, limited story | Readable story with well-chosen visuals | Carefully designed visuals supporting narrative | Convincing story with carefully designed visuals |
| **Discussion** | No awareness of limitations | Mentions limitations without implications | Identifies key limitations and their implications | Methodological + policy arena limitations discussed | Limitations discussed with suggestions for future work |
| **Conclusions** | Inconsistent, not linked to problem owner | Trivial conclusions | Consistent conclusions, appropriate advice | Advice appropriate for problem owner and multi-actor context | Convincing conclusions with advice appropriate for multi-actor context |

---

## Part 2 — Political reflection on the COP31 debate (30 %)

### Goal

Show your understanding of the different roles that (model-based) knowledge plays in decision making, drawing on the concrete negotiations of the COP31 debate.

### What to write

Analyse the different ways in which knowledge was used in the negotiation process, and write advice for actors on how model-based knowledge can appropriately support their decision-making.

You need to:
- Use academic literature from this course with correct APA references
- Account for modelling choices driven by your actor's position (politics in knowledge production)
- Use notes and examples from the negotiation process to illustrate the roles of knowledge

### Proposed outline *(you may deviate)*

**Introduction**
- The question you will answer
- Brief conceptualisation of roles knowledge can play in decision making (with references)
- How you analysed these roles in the negotiations (note-taking, grouping observations)

**Systematic analysis**
Different roles of knowledge in the negotiations, with concrete examples and consequences for the overall decision-making process.

**Recommendations**
Advice to actors participating in negotiations on how to use knowledge in different ways.

**Length:** Minimum 2,000 words, maximum 5,000 words.

### Rubric — Part 2 (10 points total)

| Criterion | Points | Description |
|-----------|--------|-------------|
| Conceptualising roles of knowledge | 1 | Three or more distinct roles identified, drawing on course literature, specific to the policy context |
| Analysing roles of knowledge in negotiations | 3 | Specific analytical steps to systematically illustrate how knowledge was used; impact on decision making clearly described |
| Account of modelling choices and actor interests | 2 | Concrete description of choices made in the modelling process and why, linked to literature on roles of knowledge |
| Recommendations for better use of knowledge | 2 | Specific strategies linked to identified challenges, grounded in course literature and specific to the policy advice |
| Reflection on risks of proposed strategy | 2 | 3+ risks addressed in detail, with justifications and potential adaptations |

---

## Submission

Submit via **Brightspace** by **19 June 2026, 17:00**:
1. Final report as PDF
2. Code repository link or ZIP file

Name your files: `EPA141A_Group<N>_FinalReport.pdf`
