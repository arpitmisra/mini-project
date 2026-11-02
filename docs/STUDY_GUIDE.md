# Study guide: rank, SVD, LoRA/PEFT, Swin and weakly supervised OCT segmentation

Prepared for the discussion on 2 November 2026. This document separates mathematical facts, established methods and proposed experiments. The current default code extracts frozen pretrained DINOv2 features. It does **not** yet implement LoRA, Swin segmentation, pseudo-label training or a validated lesion segmenter. The earlier MLX self-supervised adaptation is a separate experiment.

For today's meeting read sections 1, 7, 9 and 12 first. Then work through the numerical examples. The dependency order is:

**Vectors → rank → SVD → low-rank updates → PEFT/LoRA → patch features and attention → weak-label objectives → pseudo masks → evaluation.**

Three separate questions matter: (1) Does the software reproduce a feature tensor? (2) Does it encode useful OCT information? (3) Can it delineate lesions accurately? Success on the first is not evidence that all three are solved.

## 1. Formulate the problem before choosing a model

For image $i$, let $X_i\in\mathbb R^{H\times W\times C}$ be the scan and $y_i$ its image diagnosis. Training observes $(X_i,y_i)$, but not a pixel map $M_i\in\{0,\ldots,K\}^{H\times W}$. We want a model of:

$$
P_\theta(k\mid X_i,u,v),
\qquad
\hat M_i(u,v)=\arg\max_k P_\theta(k\mid X_i,u,v).
$$

**Everyday example:** a bag of apples has a sticker saying “contains a bruised apple.” The sticker identifies neither the apple nor the outline of its bruise. An image label is the sticker; segmentation asks for the bruise boundary.

### Why this is underdetermined

In a simplified binary formulation:

$$
y_i=\max_{u,v}M_i(u,v).
$$

One diseased pixel, the true lesion, and an entirely positive image all satisfy $y_i=1$. Image-label supervision alone cannot identify which mask is correct. This is an **identifiability problem**: distinct hidden explanations yield the same observed label.

Pretrained features, anatomy, local similarity and consistency across scans can provide useful biases, but those are additional information or assumptions. A low classification loss does not prove correct boundaries.

### Agree on what “diseased area” means

An image diagnosis does not automatically define a pixel class. Decide whether the goal is any abnormal tissue, a specified lesion type, several lesion types, or a region associated with the image diagnosis. Decide whether image labels are mutually exclusive or multilabel. Do not copy the image diagnosis onto every patch.

Weakly supervised **training** can still use an independently annotated set for **evaluation**. Its masks must not enter training. If masks select thresholds/hyperparameters, that set is validation; reserve a different test set. Without any spatial reference, report qualitative observations or localization proxies, not measured precise segmentation.

## 2. Vectors and matrix multiplication

A vector is an ordered list of measurements. For a shopping basket, $x=[\text{apples},\text{oranges}]^T$. A price vector $w=[2,3]^T$ gives the bill $w^Tx=2x_1+3x_2$. A dot product is a weighted combination.

A matrix stacks several rules. If

$$
W=\begin{bmatrix}2&3\\4&1\end{bmatrix},
$$

then $Wx$ gives two summaries. A neural linear layer does the same operation, $z=Wx+b$. An $m\times n$ matrix maps an $n$-dimensional input into an $m$-dimensional output. Always write shapes before manipulating formulas.

A linear combination is $a_1v_1+\cdots+a_rv_r$. Vectors are independent when the only combination yielding zero has all coefficients zero. Today's sales, twice today's sales and three times today's sales are three measurements of one independent quantity.

## 3. Rank of a matrix

Rank is the dimension of a matrix's column space, equivalently its row space: the number of independent directions represented by it.

$$
\operatorname{rank}(A)\leq\min(m,n).
$$

Rank is not the number of nonzero entries, classes, parameters or “important features.”

### Example: rank one

$$
A=\begin{bmatrix}1&2&3\\2&4&6\\3&6&9\end{bmatrix}
=uv^T,\qquad u=v=[1,2,3]^T.
$$

Every row and column is a multiple of one direction, so rank is 1.

**Daily analogy:** several price lists differ only by currency conversion. They contain many numbers but one underlying price pattern. Adding currencies does not add independent patterns.

### Example: calculate rank

$$
B=\begin{bmatrix}1&2&3\\2&4&6\\1&1&2\end{bmatrix}.
$$

Subtract twice row 1 from row 2, and row 1 from row 3:

$$
\begin{bmatrix}1&2&3\\0&0&0\\0&-1&-1\end{bmatrix}.
$$

Two independent nonzero rows remain, so rank is 2.

The null space consists of $x$ such that $Bx=0$. For example $x=[-1,-1,1]^T$. Rank–nullity gives:

$$
\operatorname{rank}(B)+\dim\operatorname{null}(B)=3.
$$

The null space has dimension 1. Inputs differing by that direction become indistinguishable after multiplication by $B$.

### Geometry

A rank-2 map from 3D can flatten a volume onto a plane; a rank-1 map onto a line. Directions in the null space disappear. A full-rank square matrix is invertible; a rank-deficient one loses information.

### Rank in this project

One scan has $F\in\mathbb R^{256\times384}$: rows are patches, columns are feature channels. Its rank is at most 256. Subtract the mean row and rank is at most 255, because centered rows sum to zero.

This does not prove the backbone globally needs only 256 channels; it is a bound for one scan's feature matrix.

Keep two matrices distinct:

- Feature matrix $F$: what the model outputs on data.
- Weight update $\Delta W$: how adaptation changes a layer.

Their ranks answer different questions. A rank-8 update does not require all output features to have rank 8.

### Numerical and effective rank

Floating-point noise can make tiny singular values nonzero. Numerical rank counts values above a declared tolerance:

$$
r_\epsilon=\#\{j:\sigma_j>\epsilon\}.
$$

State whether the tolerance is absolute or relative. An “effective rank” based on retained squared singular values is another chosen summary, not a universal integer property.

A sharp feature spectrum may show redundancy, background domination or collapse. It does not automatically determine an optimal LoRA rank.

## 4. Singular value decomposition

SVD factors any real $m\times n$ matrix:

$$
A=U\Sigma V^T.
$$

Full SVD has $U\in\mathbb R^{m\times m}$, $V\in\mathbb R^{n\times n}$, rectangular diagonal $\Sigma\in\mathbb R^{m\times n}$, and ordered nonnegative singular values $\sigma_1\geq\sigma_2\geq\cdots$. The columns of $U,V$ are orthonormal. Reduced SVD keeps $\min(m,n)$ directions; compact rank-$r$ SVD keeps only positive ones.

### Rotate, stretch, rotate

For $Ax=U\Sigma V^Tx$, $V^T$ changes to special input directions, $\Sigma$ scales them, and $U$ expresses the result in output directions. Zero scales erase directions.

**Daily analogy:** describe a journey along a city's main avenues rather than north/east; scale costs differently along each avenue; translate back into the map coordinates. A good coordinate system exposes the simplest transformation.

### Connection to eigenvectors

$$
A^TA=V\Sigma^T\Sigma V^T.
$$

Right singular vectors are eigenvectors of $A^TA$; $\sigma_j=\sqrt{\lambda_j(A^TA)}$. For $\sigma_j>0$, $u_j=Av_j/\sigma_j$. This explains SVD conceptually. Numerically, computing $A^TA$ squares the condition number; use a stable SVD implementation rather than doing this manually.

### Complete example

Take

$$
A=\begin{bmatrix}3&1\\1&3\end{bmatrix},\quad
v_1=\frac{[1,1]^T}{\sqrt2},\quad
v_2=\frac{[1,-1]^T}{\sqrt2}.
$$

Then $Av_1=4v_1$ and $Av_2=2v_2$. Thus $U=V=[v_1,v_2]$, $\Sigma=\operatorname{diag}(4,2)$, and

$$
A=4v_1v_1^T+2v_2v_2^T
=\begin{bmatrix}2&2\\2&2\end{bmatrix}
+\begin{bmatrix}1&-1\\-1&1\end{bmatrix}.
$$

The first pattern captures shared movement; the second captures differences.

### Truncated SVD

$$
A_k=\sum_{j=1}^{k}\sigma_j u_jv_j^T
$$

is a best rank-at-most-$k$ approximation in Frobenius norm:

$$
\|A-A_k\|_F^2=\sum_{j>k}\sigma_j^2,
\qquad
\|A-A_k\|_2=\sigma_{k+1}.
$$

Our rank-1 approximation is the all-2 matrix. Frobenius error is 2; squared error is 4. Retained squared-singular-value fraction is $16/(16+4)=80\%$. This “energy” is algebraic, not 80% diagnostic information or accuracy.

### PCA and dense-feature visualization

For centered data $Z=U\Sigma V^T$, the first $k$ columns of $V$ are PCA directions; scores are $ZV_k=U_k\Sigma_k$.

Project 384-dimensional patch features onto three components and map them to RGB for visualization. Similar colors indicate similarity in this projection, not disease identity. Component signs can flip, repeated-value subspaces can rotate, and independently fitting each image makes colors incomparable.

For fair comparison, fit PCA and its mean on training data and apply the same transform to held-out images. Test-data fitting is transductive and must be declared. Sample across patients rather than letting one scan dominate. High explained variance need not retain subtle lesion signals.

For more linear algebra see [MIT's SVD lesson](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-29-singular-value-decomposition/). The numerical examples here are worked specifically for this guide.

## 5. DINOv2, patch tokens and attention

DINOv2 is a visual representation model, not a lesion segmenter. Our ViT-S/14 checkpoint uses 14-pixel patches, 384-dimensional tokens and 12 transformer blocks. At 224×224 input, there are 16×16=256 patch tokens plus CLS; this checkpoint has no register tokens. [Meta model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)

A 14×14 RGB patch has 588 values. A learned projection maps it to 384 channels. Position information and CLS are added before transformer processing. Replicating grayscale OCT into RGB meets the expected input format but does not create three independent measurements.

### Attention mathematics

Let $X\in\mathbb R^{N\times d}$ contain token rows. One head computes:

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
\quad A=\operatorname{softmax}_{row}(QK^T/\sqrt{d_h}),\quad Y=AV.
$$

Entry $A_{ij}$ weights token $j$'s value when updating token $i$. Multiple heads learn different relationships; their outputs are concatenated and projected. Residual connections and an MLP further transform the tokens. Scaling by $\sqrt{d_h}$ moderates growing dot-product magnitudes and softmax saturation under typical assumptions.

**Daily analogy:** in a study group, each person asks a question (query), advertises expertise (key) and contributes an explanation (value). Their final notes combine relevant explanations from several people.

### Dense is not pixel-perfect

The output $F\in\mathbb R^{256\times384}$ can be reshaped to 16×16×384. Location (7,11) is token $7\cdot16+11=123$. Its anchor is a patch but attention can incorporate information from the whole crop.

Upsampling a 16×16 score map does not recover boundaries that were never resolved. CLS, patch descriptors, attention weights, feature norms, PCA colors and disease masks are different quantities. None should silently substitute for another.

The current preprocessing center-crops the scan. A segmentation project needs to revisit full-image coverage, resolution, tiling and coordinate transforms. Keep preprocessing equal when attributing differences to adapters.

### Three different experiments

DINOv2 learns representations through a self-supervised teacher–student recipe with image- and patch-level objectives and other design choices. This is a useful initialization, not proof of OCT segmentation performance. [DINOv2 paper](https://arxiv.org/abs/2304.07193)

1. **Frozen pretrained baseline:** only inference, implemented as the default now.
2. **Self-supervised OCT adaptation:** learn from unlabeled OCT appearance; the earlier MLX experiment does a simplified version.
3. **Weakly supervised segmentation:** use image diagnoses to learn spatial predictions through a stated objective; not yet implemented.

## 6. Parameter-efficient fine-tuning (PEFT)

PEFT is the category of adapting a model through relatively few trainable parameters; it is also the name of a software library. LoRA is one PEFT method. [PEFT documentation](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)

**Analogy:** rather than rebuilding a sophisticated camera, adjust a few controls or attach a small correction module. This can be efficient, but the chosen correction can still be wrong.

| Method | What learns | Main implication |
|---|---|---|
| Frozen backbone plus head | New task head | Cheap baseline; cached features are reusable |
| Full fine-tuning | Most/all backbone weights | More flexible and larger optimizer state; forgetting is possible |
| LoRA | Low-rank updates to selected maps | Small trainable state and potentially mergeable updates |
| Bottleneck adapters | Small residual networks | Nonlinear correction, often extra inference operations |
| Prompt tuning | Learned input/intermediate tokens | Alters conditioning; preserve spatial-token indexing |
| Bias-only adaptation | Selected biases | Tiny parameter count but limited flexibility |

A bottleneck adapter can be $h'=h+W_{up}\phi(W_{down}h)$. The middle dimension is small. Nonlinear $\phi$ makes this different from one low-rank linear update.

Fewer trainable weights often reduces gradient/optimizer memory. It does not eliminate the frozen backbone, forward computation or activation/backpropagation cost. Measure time, peak memory and trainable parameters separately.

**Caching rule:** final cached features can train a downstream head with a frozen backbone. They cannot train internal LoRA weights, because the features must change with those weights and gradients must flow through the backbone. After adaptation, regenerate the cache with the adapter identity included.

## 7. LoRA-based adapters on DINOv2

LoRA freezes $W_0$ and learns:

$$
W=W_0+\Delta W,\quad \Delta W=sBA,\quad s=\alpha/r,
$$
$$
A\in\mathbb R^{r\times d_{in}},\quad
B\in\mathbb R^{d_{out}\times r},\quad
h=W_0x+sB(Ax).
$$

The **update** has rank at most $r$; the complete weight can remain full rank. This is not necessarily a low-rank approximation of the original weight. [LoRA paper](https://arxiv.org/abs/2106.09685)

### Intuition

A restaurant adapting to a new cuisine may change a few coordinated dimensions—spice, texture, sweetness—rather than rewrite every recipe independently. $A$ identifies a few adjustment coordinates; $B$ distributes them to the full output. This illustrates the constraint, not evidence that every OCT adaptation is intrinsically low rank.

### Parameter arithmetic

For a 384×384 projection:

$$
\#W_0=384^2=147456,\qquad
\#(A,B)=r(384+384).
$$

At rank 8: 6144 trainable parameters, 4.17% of that projection. Query and value in all 12 blocks give:

$$
12\times2\times6144=147456
$$

adapter parameters, excluding any new head. Clearly state whether a percentage uses targeted matrices or the entire model as its denominator.

Rank 8 means at most eight independent **linear update directions per targeted map**, not eight lesions, patch locations or clinical classes.

### Selecting modules

Query/value projections are a sensible first experiment; key/output/MLP projections can be ablations. Module names differ: some implementations separate Q/K/V, others fuse them. Inspect names and shapes rather than blindly copying a target pattern from another model. The [PEFT vision example](https://huggingface.co/docs/peft/v0.7.0/task_guides/image_classification_lora) illustrates targeting projection modules, not universal DINOv2 naming.

### Initialization and gradients

A common initialization is random $A$, zero $B$, yielding zero initial update and the original pretrained function. [Official LoRA implementation](https://github.com/microsoft/LoRA)

For one input, let $g=\partial L/\partial h$:

$$
\frac{\partial L}{\partial B}=s\,g(Ax)^T,\qquad
\frac{\partial L}{\partial A}=s\,B^Tg\,x^T.
$$

When $B=0$, the first gradient of $A$ is zero, but $B$'s can be nonzero. After $B$ moves, $A$ receives gradients. Initializing **both** factors to zero would make both gradients zero in this formulation.

The objective still matters. LoRA only specifies which changes are allowed; a loss specifies the desired behavior. Candidate OCT objectives include image-label MIL, self-supervised consistency or later confidence-masked pseudo-label segmentation.

### Relationship to SVD

SVD decomposes a **known** matrix. LoRA learns factors from the loss; it does not ordinarily take an SVD of $W_0$. If a known update has compact SVD $U_r\Sigma_rV_r^T$, choose $B=U_r\Sigma_r^{1/2}$, $A=\Sigma_r^{1/2}V_r^T$, ignoring the separate scale convention. Training does not know this update beforehand.

Factorization is nonunique: $BA=(BR)(R^{-1}A)$ for any invertible $R$. Individual factor entries are not unique explanations.

### Compare and save responsibly

Compare frozen DINOv2 plus a fixed weak-label head against LoRA-DINOv2 plus the same head, labels, loss and split. Otherwise a gain may come from another change.

Save exact base revision, target modules, rank, scale, adapter and task-head weights, preprocessing and training metadata. Compatible linear updates can be merged as $W_0+sBA$; test merged/unmerged numerical agreement. The base weights are still needed.

The present code does not claim to implement this LoRA experiment. Low training loss would not establish lesion accuracy.

## 8. Swin Transformers

Swin uses local-window attention, shifted partitions and hierarchical features. This is a different way to organize spatial computation from DINOv2's plain ViT backbone. [Swin paper](https://arxiv.org/abs/2103.14030)

### Why windows help

**Analogy:** at a large workshop, discussions occur at tables rather than between every pair of participants. Then table boundaries shift so neighboring groups exchange information.

For $N$ tokens and $M\times M$ windows, each window has $M^2$ tokens and there are approximately $N/M^2$ windows. Token-pair interactions scale as:

$$
(N/M^2)(M^2)^2=NM^2
$$

instead of $N^2$. With projection costs:

$$
\Omega(\mathrm{MSA})\approx4NC^2+2N^2C,\qquad
\Omega(\mathrm{W\!-\!MSA})\approx4NC^2+2M^2NC.
$$

For fixed window size and channels, the interaction term is linear in token count. This does not mean the entire model is cost-free or constant-time.

### Shift, mask, position

A subsequent block shifts the window partition, commonly by roughly half a window. Efficient cyclic shifts require masks to prevent wrapped image-edge tokens becoming false neighbors. Padding, window partition and reversal must align correctly.

Relative-position bias can enter as:

$$
\operatorname{softmax}(QK^T/\sqrt{d_h}+B_{relative}+M_{mask})V.
$$

It encodes displacement inside a window.

### Hierarchy

Patch merging groups 2×2 neighbors: concatenate into 4C channels, then commonly project to 2C. Height and width halve; token count quarters. A conventional patch-size-4 Swin on 224×224 input can produce 56×56, 28×28, 14×14 and 7×7 stages. Dimensions depend on the variant.

This pyramid offers fine detail and coarse context to a decoder. DINOv2 ViT-S/14 keeps its patch grid fixed through blocks. Neither alone defines a lesion mask.

### Must we build Swin from scratch? No.

The [official Microsoft repository](https://github.com/microsoft/Swin-Transformer) and [Torchvision model builders](https://docs.pytorch.org/vision/stable/models/swin_transformer.html) already provide implementations/pretrained options.

Distinguish:

1. Implementing the architecture as a learning exercise.
2. Training from random initialization.
3. Loading an existing pretrained Swin.
4. Developing a task-specific decoder/objective.

The fourth may be the actual research need. A new MLX port or a new architecture could require extra implementation, but the baseline does not require it. Comparing random Swin to pretrained DINOv2 confounds architecture with pretraining.

## 9. Turning image labels into spatial supervision

This is a proposed research path, not current repository functionality.

### Multiple-instance learning (MIL)

Treat an image as a bag of descriptors $F=\{f_1,\ldots,f_N\}$. A patch scorer produces evidence logits $a_{jc}$ for class $c$ at patch $j$. Pool them to image logits:

$$
z_c=\frac1N\sum_j a_{jc}\ \text{(mean)},\qquad
z_c=\max_j a_{jc}\ \text{(max)},
$$
$$
z_c=\frac1\beta\log\left(\frac1N\sum_j e^{\beta a_{jc}}\right)
\ \text{(smooth maximum)}.
$$

Large $\beta$ approaches max; small $\beta$ approaches mean. Implement log-sum-exp stably by subtracting the maximum.

Max can focus on one highly discriminative patch; mean can dilute a small lesion. Pooling therefore encodes a spatial assumption.

For mutually exclusive diagnoses:

$$
p_c=\frac{e^{z_c}}{\sum_k e^{z_k}},\qquad
L_{\mathrm{image}}=-\log p_y.
$$

For multilabel images, use independent sigmoids and binary cross-entropy rather than one softmax. Preserve the original diagnoses unless a binary experiment is explicitly intended.

### Attention pooling

A learned aggregation can use:

$$
\alpha_j=
\frac{\exp(w^T\tanh(Vf_j))}
{\sum_\ell\exp(w^T\tanh(Vf_\ell))},
\qquad b=\sum_j\alpha_jf_j.
$$

A head predicts the image label from $b$. Attention-MIL provides a model of trainable bag aggregation; its weights are not automatically lesion probabilities. [Attention-based MIL](https://arxiv.org/abs/1802.04712)

**Daily analogy:** a quality inspector prioritizes a few suspicious apples to decide whether a box is defective. Their attention identifies evidence, not necessarily every damaged area.

### Why an image-classification loss can appear in a segmentation project

The final deliverable can be segmentation while classification is an intermediate training signal. Avoiding a standalone binary-classifier deliverable does not prohibit image-label supervision mathematically.

But a classifier reading only CLS does not directly define a justified dense mask. The objective must connect image labels to spatial scores or another explicitly tested localization mechanism.

### CAM algebra and limitations

With a linear class score on globally averaged feature channels:

$$
z_c=\sum_k w_{ck}\left(\frac1N\sum_j F_{jk}\right)+b_c,
$$

the spatial contribution is $H_c(j)=\sum_k w_{ck}F_{jk}$. This is the basic class-activation-map construction. [CAM paper](https://arxiv.org/abs/1512.04150)

Strong evidence may highlight a discriminative fragment, scanner artifact or correlated background. It need not cover the full lesion. Scaling a heatmap to [0,1] does not calibrate it as probability. Thresholding must be specified and independently evaluated.

### Pseudo-labels with an ignore band

For a spatial score defined by a declared procedure:

$$
\tilde M_j=
\begin{cases}
1 & s_j\ge t_{\mathrm{high}},\\
0 & s_j\le t_{\mathrm{low}},\\
\mathrm{ignore} & \text{otherwise}.
\end{cases}
$$

Train a decoder only on the confident set $\mathcal C$:

$$
L_{\mathrm{pseudo}}=
-\frac1{|\mathcal C|}
\sum_{j\in\mathcal C}\log P_\theta(\tilde M_j\mid X,j).
$$

Handle an empty confident set explicitly. Unknown patches should not automatically be treated as background. The general seed/expand/boundary-constraint strategy is established in weakly supervised segmentation, but an OCT version requires separate evidence. [Seed, Expand and Constrain](https://arxiv.org/abs/1603.06098)

### Consistency and smoothness

Geometric consistency encourages $P(TX)\approx T(P(X))$. Correct coordinate transforms and valid-pixel masks are essential. Different crop locations cannot be compared without correspondence.

A feature-affinity regularizer might use:

$$
w_{ij}=\exp(-\|f_i-f_j\|^2/\tau),\qquad
L_{\mathrm{smooth}}=\sum_{(i,j)}w_{ij}\|p_i-p_j\|^2.
$$

Choose neighborhood and normalization deliberately. Too much smoothness can erase small lesions or spread false positives. Anatomy constraints help only when their assumptions match the target.

A proposed combined loss:

$$
L=L_{\mathrm{image}}+\lambda_pL_{\mathrm{pseudo}}
+\lambda_cL_{\mathrm{consistency}}+\lambda_sL_{\mathrm{smooth}}.
$$

This is a design template, not a validated recipe. Add one component at a time and ablate it. Wrong pseudo-labels can reinforce themselves; confidence filtering and teacher averaging do not prove correctness.

### A practical sequence

1. Establish frozen-feature extraction and provenance.
2. Train a simple patch/MIL head on image labels.
3. Inspect spatial evidence and identify shortcuts/failure cases.
4. Produce confidence-filtered pseudo masks with explicit ignored regions.
5. Train a segmentation decoder using those targets.
6. Compare a LoRA backbone using the same objective/decoder and protocol.
7. Evaluate against independent spatial references.

Some steps may be combined in a model, but keeping the baseline experiments separate helps diagnose where improvements or failures arise.

## 10. Evaluation and fair claims

Relative to reference masks:

$$
\mathrm{Dice}=\frac{2TP}{2TP+FP+FN},\qquad
\mathrm{IoU}=\frac{TP}{TP+FP+FN}.
$$

Define empty-mask conventions. Report per-class and per-case distributions. Background can dominate pixel accuracy. Boundary distances can be useful, but physical units require reliable pixel-spacing information.

Image classification accuracy/AUC answers a different question. It cannot establish segmentation accuracy without spatial references. Blinded expert review offers evidence but is not interchangeable with reference-mask evaluation.

### Leakage controls

- Split by patient and, where appropriate, eye/volume/site.
- Different filenames or directories do not establish independence.
- Related slices must not leak across splits.
- Fit PCA, prototypes, learned preprocessing and thresholds on training/validation only.
- Do not adapt on test images while claiming an inductive evaluation.
- Audit exact and near duplicates.
- For correlated scans, estimate uncertainty at the patient rather than pixel level.

The existing loader checks exact duplicate bytes; it does not establish all these conditions.

### Controlled comparison plan

| Experiment | Backbone | Same spatial head/objective? | What it asks |
|---|---|---|---|
| A | Frozen pretrained DINOv2 | Reference head | Baseline |
| B | DINOv2 + LoRA | Same as A | Does efficient adaptation help? |
| C | Fully fine-tuned DINOv2 | Same as A | Accuracy/resource tradeoff against LoRA |
| D | Pretrained Swin | Match capacity/protocol where feasible | Does hierarchy help? |
| E | Best controlled backbone | Change one objective component | Which loss component contributes? |

Keep evaluation patients, label ontology, preprocessing coverage, budget and threshold selection comparable. If multiple ingredients change, call it a system comparison rather than a clean LoRA ablation.

Ranks 2,4,8,16 could be trial values; they are not established OCT optima. Use repeated seeds when feasible and report failures. Improvements are a hypothesis, not a promised result.

## 11. Public research artifact checklist

A release should include the short quick start, real-model self-test, exact checkpoint identity, tested-platform records, data instructions, permitted split manifests, feature schema/coordinates, configurations, seeds, evaluation scripts, and time/memory/storage measurements.

Capture dependency versions. Version the base model and adapter separately. Distinguish baseline features from fine-tuned ones. Describe crop limitations and failure cases. Add citation metadata when real paper details exist.

Public availability does not automatically settle code/data/model licensing. Review actual terms and choose a new-code license with the owner before calling the release open source; do not invent permissions to redistribute patient data or weights. The upstream checkout retains its own license.

Reproducibility need not mean identical floating-point bits across devices. Declare numerical tolerances and the statistical evaluation protocol.

## 12. What to say in today's meeting

“We separate a reproducible pretrained feature baseline from adaptation research. Next we must define the lesion target, establish spatial evaluation, and choose an image-label objective that creates spatial evidence. We can compare LoRA against a frozen backbone under the same objective, then test Swin or decoder changes. We will not treat a feature tensor or classification heatmap as already-validated precise segmentation.”

Questions to resolve:

1. Exactly which pixels/lesion types should be segmented?
2. Are image labels single-label or multilabel? Do we have trustworthy patient/volume IDs?
3. Can we obtain a small expertly annotated validation/test set or compatible external benchmark?
4. Is frozen DINOv2 plus a weak-label spatial head the agreed baseline?
5. Is LoRA the first adaptation experiment?
6. Is Swin a backbone comparison, a decoder component or an educational exercise?
7. What metrics, compute budget and ablations define success?
8. What artifacts may be released, and under what terms?

## 13. Check your understanding

1. Every column is a multiple of one nonzero vector. Rank? **1.**
2. Can a 256×384 feature matrix have rank 384? **No.**
3. Does rank-8 LoRA force the entire updated weight to rank 8? **No, only its update.**
4. Does 95% PCA variance imply 95% lesion information? **No.**
5. Can a perfect image classifier have a poor mask? **Yes.**
6. Can cached original final features train internal LoRA weights? **No.**
7. Does interpolating a patch grid yield verified precise boundaries? **No.**
8. Must Swin be rewritten or trained from random initialization? **No.**

Run the companion mathematical examples:

```bash
.venv-dense/bin/python scripts/learn_linear_algebra.py
```

These verify rank, SVD, approximation error, LoRA parameter counts and a low-rank update to a full-rank weight. They are teaching examples, not OCT experiments.
