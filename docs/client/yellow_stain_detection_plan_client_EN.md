# Project Plan: Vision Model for Detecting Yellow Stains on Foam Bra Cups

> Version: v1 ｜ Date: ＿＿＿＿
>
> This plan will be refined based on the materials and confirmations provided by the Client.

---

## 1. Background and Objectives

The Client's existing black-dot defect detection system is based on YOLO, with a recognition rate already above 99%, and comes with an annotation and batch-inference platform. This project builds on that foundation to add the ability to detect **yellow stains** — a yellowish discoloration that is more subtle than black dots (it requires careful inspection by eye, is irregular in shape, and has blurred edges).

**Objective**: Deliver a trained vision model that reliably detects yellow stains and classifies products as conforming (pass) or defective.

**Duration**: 6–7 weeks.

---

## 2. Deliverables and Scope

**Core deliverable**
- One trained model file in `.evo` format meeting the agreed metrics.

**Supporting deliverables (to help the Client's technical team continue and reproduce the work)**
- Annotated dataset；
- Training code；
- Technical documentation (method, annotation criteria, metrics, reproduction steps).

**Out of scope (handled by the Client's technical team)**
- Production deployment, batch operation, and the interface / platform of the model.

Because the core deliverable is a model file, project success is primarily reflected in **model recognition performance**. For this reason, **aligning on acceptance criteria up front** is critical to the project (see Sections 5 and 8).

---

## 3. Technical Approach

**Overall direction**: The work follows the same YOLO / Ultralytics paradigm as the existing system, so as to align with the existing annotation workflow and data format and to reuse a mature toolchain.

**Important note — no neural network is built from scratch**: This project uses a mature open-source framework and **fine-tunes its pretrained models (transfer learning)** rather than writing a network from the ground up. Pretrained models have already learned fundamental visual capabilities — edges, textures, shapes — from large volumes of general images; we only need to fine-tune them on the project's samples. This is precisely what makes the project achievable within 6–7 weeks using a few hundred images.

**Model and algorithm selection**

| Purpose | Selection | Notes |
|---------|-----------|-------|
| Primary model | **Ultralytics YOLO** (current stable version, e.g. the YOLO11 family; the `-seg` variant for segmentation) | Same lineage as the existing black-dot system, mature toolchain; fine-tuned from pretrained weights |
| Model form | **Segmentation** preferred, with rectangular detection or classification as needed | Yellow stains are irregular patches; segmentation fits their actual shape better than a bounding box |
| Baseline method | **Classical color thresholding (OpenCV, based on LAB / HSV)** | Fast, interpretable, may resolve part of the problem directly; serves as a baseline reference |
| Fallback | **Anomaly detection (Anomalib, e.g. PatchCore / PaDiM)** | Trains on conforming samples only, low demand for defect samples; suited to cases with few samples or a weak signal |
| Classification | **Classification CNN** (e.g. ResNet / EfficientNet) | Judges a region or whole image as conforming / defective; can serve as one stage in the pipeline |
| Annotation aid | **SAM / SAM 2** | Auto-generates a contour from a click, greatly speeding up segmentation annotation |
| Preprocessing | **LAB color-space enhancement** | Enhances the yellow-stain signal, improving both recognition and annotation (see below) |

Which route is ultimately adopted depends on the separability assessment in Section 6 (a clear signal favors segmentation; few samples or a weak signal favors anomaly detection). Mature industrial inspection systems are typically a combination of several models and classical methods; this project will, after understanding the composition of the existing detection pipeline (see Section 5), choose a structure that integrates with it, using a classification CNN or similar as one stage where appropriate.

**Key method — color-space enhancement (LAB)**: The difference between yellow and white is fundamentally a difference in color. After converting an image to the LAB color space, its b channel (the blue–yellow axis) can significantly amplify a yellow-stain signal that is hard to see by eye, making it far easier to recognize and annotate. This is the core verification method in the first phase and can also serve as model input preprocessing.

**On attainable metrics**: Yellow stains are more subtle than black dots, so **the final achievable recognition metric depends on the separability of yellow stains in the imaging**. The project will complete a separability assessment and fix the technical route in the first phase (Week 1). We recommend that both parties jointly confirm the **target metric** after that assessment (see Section 6).

---

## 4. Project Plan and Milestones

**Execution principles**
1. **End-to-end first**: Get the full "annotation → training → inference" pipeline running as early as possible, then iterate on performance.
2. **Inputs up front**: Raise the required materials and confirmations (Section 5) at the very start of the project, to avoid critical dependencies causing blockers.
3. **Controlled iteration**: Time-box the core tuning phase and drive decisions by metrics.
4. **Reproducible**: Version-control code and experiment records throughout.

**Milestones at a glance**

| Phase | Timing | Key milestone |
|-------|--------|---------------|
| 1 | Week 1 | Separability assessment done, technical route fixed, target-metric consensus |
| 2 | Week 2 | Annotation guidelines finalized, end-to-end pipeline verified |
| 3 | Week 3 | Annotation complete, baseline model and baseline metrics |
| 4 | Week 4 | Model performance significantly improved |
| 5 | Week 5 | Target metric reached or approached |
| 6 | Week 6 | Deliverable `.evo` and delivery package ready |
| 7 | Week 7 | Edge cases handled, documentation finalized, formal delivery |

**Phase details**

**Phase 1 (Week 1) — Separability assessment and route selection**
- Complete a **separability assessment** of yellow stains via LAB / color-space methods, and fix the technical route；
- Survey the sample data (stain size, contrast, occurrence rate, distribution of hard cases)；
- Align with the existing system: annotation tool, data format, `.evo` generation method；
- Set up the development and training environment.
- **Milestone**: separability conclusion + technical route fixed + (recommended) target-metric consensus.

**Phase 2 (Week 2) — Annotation guidelines and end-to-end pipeline**
- Draft **annotation guidelines** with positive and negative examples, clarifying the judgment and outlining criteria；
- Configure the annotation tool (including SAM-assisted segmentation for efficiency), with data export aligned to the training format；
- Use a small sample to get the full "annotation → training → batch inference" pipeline running (the goal is to connect the flow end to end).
- **Milestone**: annotation guidelines finalized + end-to-end pipeline verified.

**Phase 3 (Week 3) — Data annotation and baseline model**
- Complete the bulk of the annotation, calibrating the criteria continuously as work proceeds；
- Split into training / validation sets (the acceptance test set is kept out of training)；
- Train the first model version and establish baseline metrics and an experiment log.
- **Milestone**: annotation complete + baseline model and baseline metrics.

**Phase 4 (Week 4) — Model iteration (1)**
- Diagnose the baseline performance and pinpoint the main contributing factors；
- Preprocessing (including LAB enhancement), data augmentation, hyperparameter tuning；
- Introduce color-thresholding fusion or an anomaly-detection approach as needed.
- **Milestone**: model performance significantly improved over the baseline.

**Phase 5 (Week 5) — Model iteration (2) and decision-threshold tuning**
- Iterate until the metric stabilizes；
- Tune the decision threshold according to the Client's relative preference between **false negatives / false positives**；
- Evaluate on the acceptance test set against the target metric.
- **Milestone**: target metric reached or approached.

**Phase 6 (Week 6) — Model export and validation**
- Export the model to `.evo` using the Client's method, with a trial export first to confirm the process and format compatibility；
- Complete final validation against the acceptance criteria；
- Assemble the delivery package (model, annotated data, training code, documentation).
- **Milestone**: deliverable `.evo` + delivery package ready.

**Phase 7 (Week 7) — Buffer and delivery**
- Handle edge cases such as unusual lighting / angles / rare morphologies；
- Finalize documentation and deliver formally.

**On timeline flexibility**: The above is laid out over 7 weeks, with Week 7 as buffer. If the duration is 6 weeks, priority will go to the core path (separability assessment, end-to-end connection, baseline and iteration, model export and validation), with iteration depth and edge-case polishing compressed accordingly; specific priorities can be confirmed with the Client at project start.

---

## 5. Materials, Confirmations, and Resources Required from the Client

To keep the project on schedule, the following are recommended to be provided or confirmed at the **start of the project**. Priorities are marked High / Medium / Low.

**(a) Materials to provide**

| Item | Purpose | Priority |
|------|---------|----------|
| An existing `.evo` file sample (e.g. from the black-dot model) and its **generation method / script** | Ensure the final delivery format is compatible with the Client's system | High |
| Sample images **confirmed to contain yellow stains** (as many as possible) | Yellow-stain feature analysis, separability assessment, model training | High |
| **Conforming** sample images (as many as possible) | The anomaly-detection approach and establishing a "normal" baseline | High |
| **Yellow-stain judgment criteria / reference cases** (ideally with expert-annotated examples) | Unify annotation and acceptance criteria, ensuring a shared understanding | High |
| Annotation data samples and format spec from the black-dot model | Align data format and annotation style | Medium |
| Access to and instructions for the existing annotation tool | Reuse the existing annotation workflow | Medium |
| The black-dot model's training configuration / experience (if convenient) | Reference to improve efficiency | Low |

**(b) Items to confirm**

| Item | Purpose | Priority |
|------|---------|----------|
| **Acceptance criteria**: test data, evaluation metric, and pass threshold | Define the goal and basis for acceptance | High |
| **Relative tolerance for false negatives vs. false positives** | Determines the direction of decision-threshold tuning | High |
| **Composition of the existing black-dot detection pipeline**: which models / methods are used (YOLO, classification CNN, classical processing, etc.), how they are chained, and which part the `.evo` corresponds to; whether yellow stains will follow the same structure | Understand the system composition for reuse, alignment, and scope definition (including whether yellow stains use detection / segmentation / classification) | Medium |
| Occurrence rate of yellow stains | Assess data distribution and approach selection | Medium |
| Scope confirmation (whether only the `.evo` is needed; whether annotated data / code is included; deployment handled by the Client) | Define scope | Medium |

**(c) Resources to provide**

| Item | Notes |
|------|-------|
| Compute | A cloud server with an NVIDIA GPU (≥16GB VRAM), or reimbursement of cloud-GPU costs. A GPU is needed during training; usage is on-demand, total usage is limited, and cost is controllable. |

---

## 6. Key Decision (Separability Assessment Result → Technical Route)

The result of the first-phase separability assessment will determine the technical route and serve as the basis for aligning on the target metric:

| Assessment result | Route |
|-------------------|-------|
| Yellow-stain signal clearly separable | Primarily a **segmentation** approach (optionally with LAB-enhanced input), with color thresholding as the baseline; performance expected to be good. |
| Signal weak / unstable | Combine multi-channel methods, color normalization, anomaly detection, etc.; and recommend re-confirming the **target metric** with the Client. |
| Few confirmed yellow-stain samples | Favor **anomaly detection** (low demand for defect samples). |
| Sufficient confirmed yellow-stain samples | Favor **supervised detection / segmentation**. |

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Limited separability of yellow stains (weak signal) | Complete the assessment in the first phase; adjust the approach as needed and align on the target metric with the Client |
| Inconsistent annotation criteria | Establish annotation guidelines with examples, aid recognition with LAB enhancement, and apply ongoing quality checks |
| Delay of key inputs (materials / confirmations) | Raise the list at project start; before samples arrive, assess with available visible samples to reduce blocking |
| `.evo` export format compatibility | Obtain a sample and the generation method early, and trial-export ahead of time to verify |
| Tight schedule | End-to-end first, controlled iteration, and scope prioritization |

---

## 8. Acceptance and Metrics

Because the core deliverable is a model file, we recommend that both parties fix the acceptance criteria at the **start of the project**, including:

- A clear **test dataset, evaluation metric, and pass threshold**；
- The relative priority of **false negatives vs. false positives** (industrial inspection is usually more cautious about false negatives, i.e. it leans toward "better a false alarm than a missed defect").

The project will keep an independent test set for objective evaluation; if the Client provides an acceptance test set, it will be used as the basis for final validation.

---

*This is an initial version. The final approach, target metric, and schedule will be refined and fixed jointly, based on the materials and confirmations provided by the Client.*
