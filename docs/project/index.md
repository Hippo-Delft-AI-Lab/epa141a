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
| 2 | Political reflection on the climate change debate | **30 %** |

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

## Part 2 — Reflection on the use of (model-based) knowledge in the negotiations (30 %)

Aim of this part 2 of the assignment is to show your understanding of the different roles (model-based) knowledge plays in decision making and in the concrete negotiations simulated in the debates that are needed to make the decision and create policies.

<ol type="a">
<li>You will write a reflection section as part of the report, that analyses the different ways in which knowledge has been used in the negotiation process</li>
<li>Based on this analysis, you write an advice <em>for the actors</em> that participated in the negotiation on the appropriate ways in which model-based knowledge can support their decision-making processes.</li>
</ol>

You need to

<ol type="a">
<li>Use the academic literature of this course and make appropriate references to it (use APA reference system),</li>
<li>Account for the choices you've made in the modelling that were driven by actors' positions (politics in knowledge production);</li>
<li>Use notes and examples on the negotiation process on climate mitigation to illustrate the ways in which the actors used the knowledge from the model(s) and other forms of expertise.</li>
</ol>

Proposed table of content (you may deviate):

1. Introduction
   <ol type="a">
   <li>Question you will answer in this section</li>
   <li>Brief conceptualization of what roles knowledge can play in decision making, based on academic literature and by using correct references</li>
   <li>Brief explanation of how you analyzed these roles of knowledge in the negotiations process (based on the note-taking, the grouping of your observations on the different roles of knowledge)</li>
   </ol>
2. Systematic analysis of different roles of knowledge in the negotiations/decision making with concrete examples of what consequences were of these different roles for the negotiations and the overall decision making process.
3. Recommendations to actors part of the negotiations and decision making to use knowledge in different ways during the negotiations

Part 2 of the report should be a minimum of 2000 words and a maximum of 5000 words.

**Rubric for the reflection on the use of (model-based) knowledge in the negotiations**

:::{list-table}
:header-rows: 1
:widths: 22 13 13 17 17 18

* - Criterion (Total: 10 points)
  - Insufficient
  - Weak
  - Sufficient (5.8)
  - Good (7.9)
  - Excellent (10)
* - **Conceptualising roles of knowledge in decision making/negotiations** *(1 point)*
  - Task completely or largely unaddressed. **0 pts**
  - Some roles of knowledge addressed but not described in depth. **0.25 pts**
  - Three or more distinct roles identified, but description is rather broad. **0.8 pts**
  - Three or more distinct roles identified and described well; authors draw on course literature and make these roles specific to the context and proposed policy advice. **0.9 pts**
  - Three or more distinct roles identified and described well; authors draw on course and other academic literature; roles very clearly specified to the proposed policy advice using relevant literature and good arguments. **1 pt**
* - **Analysing roles of knowledge in negotiations/decision making** *(3 points)*
  - Task completely or largely unaddressed. **0 pts**
  - Vague idea of some roles of knowledge in decision making and the negotiation process students engaged in. **0.5 pts**
  - Concrete ideas on roles of knowledge in decision making and some illustrations from the negotiations and decision making process students engaged in. **1.8 pts**
  - Specific analytical steps clearly identified and conducted to illustrate how knowledge was used in different ways in the negotiations and decision making process students engaged in. **2.5 pts**
  - Multiple specific analytical steps clearly identified and conducted to systematically analyse how knowledge was used; impact of these different roles on the decision making process clearly described. **3 pts**
* - **Account of the choices made in the modelling process, and whether they were based on the actor's interests** *(2 points)*
  - Task completely or largely unaddressed. **0 pts**
  - Vague description of choices made during the modelling process. **0.25 pts**
  - Concrete description of choices made in the modelling process. **1.2 pts**
  - Very concrete description of choices made in the modelling process and a clear explanation of why these choices were made. **1.5 pts**
  - Very concrete description of choices made in the modelling process and a clear explanation of why, linked to the academic literature on different roles of knowledge in decision making. **2 pts**
* - **Recommendations for making better use of different roles of knowledge in negotiations** *(2 points)*
  - Task completely or largely unaddressed. **0 pts**
  - Vague idea of how to make better use of different roles of knowledge in decision making/negotiations. **0.25 pts**
  - Clear idea of how technical advice can be 'improperly' used; a strategy proposed for dealing with this. **1 pt**
  - Specific strategies for behaviour in the decision-making process linked to identified challenges, based on course literature and specific to the proposed policy advice. **1.5 pts**
  - Specific strategies for behaviour in the decision-making process linked to identified challenges, based on course literature and general academic literature, and very specific to the proposed policy advice. **2 pts**
* - **Reflection on the proposed strategy: risks, gaps, and potential adaptations** *(How well have you addressed the key roles you identify? What are you missing that might also be important?) (2 points)*
  - Task completely or largely unaddressed. **0 pts**
  - Some risks addressed but not structured or clear. **0.25 pts**
  - Some risks addressed appropriately and in detail. **1 pt**
  - 3 or more risks addressed in detail, with justifications (using literature and experience in the negotiations) for why these are difficult to address in this context. **1.5 pts**
  - 3 or more risks addressed in detail, with justifications (using literature) for why these are difficult to address; includes potential adaptations and exploration of their impact. **2 pts**
* - **Total**
  - **0**
  - **1.5**
  - **5.8**
  - **7.9**
  - **10**
:::

---

## Submission

Submit via **Brightspace** by **19 June 2026, 17:00**:
1. Final report as PDF
2. Code repository link or ZIP file

Name your files: `EPA141A_Group<N>_FinalReport.pdf`
