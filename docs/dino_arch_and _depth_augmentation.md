## Multimodal Block Detection and Pose Validation

### 1. Overall architecture

The geometric point-cloud pipeline remains the primary detector and pose estimator:

[
\text{PCL}
\rightarrow
\text{geometric segmentation}
\rightarrow
\text{block candidates}
\rightarrow
\text{pose hypotheses}
]

RGB and projected-depth processing are introduced only as complementary modules for recovering failure cases and validating ambiguous poses:

[
C =
C_{\mathrm{PCL}}
\cup
C_{\mathrm{RGB}}
\cup
C_{\mathrm{depth}}
]

The final pipeline is:

[
\boxed{
\text{geometric PCL detection}
+
\text{RGB recovery}
+
\text{depth recovery}
+
\text{RGB-D pose validation}
}
]

---

## 2. RGB block-recovery module

### Purpose

The RGB module should detect blocks that were missed by the geometric PCL pipeline, for example because of:

* incomplete or noisy point clouds,
* failed plane extraction,
* partial gripper occlusion,
* merged planes or nearby objects,
* unfavorable viewing angles.

### Architecture

Use a pretrained self-supervised DINO backbone:

[
I_{\mathrm{RGB}}
\rightarrow
f_{\mathrm{DINO}}
\rightarrow
F_{\mathrm{patch}}
\rightarrow
h_{\mathrm{seg}}
\rightarrow
M_{\mathrm{block}}
]

Here:

* (f_{\mathrm{DINO}}) is a frozen DINOv2 ViT backbone,
* (F_{\mathrm{patch}}) contains spatial patch features,
* (h_{\mathrm{seg}}) is a lightweight dense-prediction head,
* (M_{\mathrm{block}}) is a block probability mask.

DINOv2 features are suitable for both image-level and pixel-level downstream tasks.

A simple per-patch MLP is appropriate as a first baseline. For better spatial resolution, use a small DPT- or FPN-style decoder that combines features from several transformer layers and upsamples them to image resolution:

[
{F^{(l_1)},F^{(l_2)},F^{(l_3)},F^{(l_4)}}
\rightarrow
\text{feature fusion}
\rightarrow
M_{\mathrm{block}}
]

DPT demonstrates this principle by reconstructing image-like feature maps from transformer tokens at multiple layers and progressively producing dense predictions.

### Conversion into 3D candidates

The predicted RGB mask is associated with the calibrated point cloud:

[
M_{\mathrm{block}}
+
D_{\mathrm{observed}}
+
K
+
T_{\mathrm{camera}\rightarrow\mathrm{world}}
\rightarrow
P_{\mathrm{RGB}}
]

The corresponding 3D points are then:

1. clustered,
2. filtered for minimum support,
3. checked against existing PCL candidates,
4. passed to the geometric pose-estimation stage.

The RGB network therefore proposes **where another block may exist**. The geometric pipeline still determines the physically consistent 3D pose whenever sufficient point-cloud data are available.

---

## 3. Projected-depth recovery module

### Motivation

A projected depth image contains the same measurements as the point cloud and therefore introduces no new sensor information. However, it presents the geometry in a regular 2D grid that can be easier for a learned detector to process.

It can recover failures caused by the handcrafted geometric algorithm, but not blocks that have no valid depth measurements.

### Input representation

Project the point cloud into the RGB camera:

[
P_{\mathrm{PCL}}
\rightarrow
D_{\mathrm{observed}}(u,v)
]

Use z-buffering so that the nearest point is stored at each pixel.

Instead of using only raw depth, construct a small geometric image:

[
X_D =
[
D,,
M_{\mathrm{valid}},,
N_x,N_y,N_z,,
H
]
]

where:

* (D): camera-relative depth,
* (M_{\mathrm{valid}}): valid-measurement mask,
* (N): estimated surface normals,
* (H): height relative to the detected ground plane.

The validity mask is important: zero depth must not be confused with a real surface.

### Network

Use a small CNN or U-Net rather than another RGB-pretrained DINO backbone:

[
X_D
\rightarrow
h_{\mathrm{depth}}
\rightarrow
M_{\mathrm{block,depth}}
]

This branch learns geometric patterns such as:

* rectangular top surfaces,
* vertical front planes,
* expected block dimensions,
* transitions between block and ground,
* partially visible block geometry.

The resulting masks are back-projected and processed in the same way as RGB proposals.

---

## 4. Pose-candidate validation

After collecting all candidates, the geometric pipeline generates one or more pose hypotheses:

[
T_1,T_2,\ldots,T_n
]

This should explicitly include ambiguous orientations such as:

[
\psi,\quad
\psi+90^\circ,\quad
\psi+180^\circ,\quad
\psi+270^\circ
]

For every hypothesis, render the known CAD model using the calibrated camera.

### Rendered candidate representation

For candidate (T_i), generate:

[
R_i =
[
I_{\mathrm{render}},
D_{\mathrm{render}},
M_{\mathrm{render}},
N_{\mathrm{render}}
]
]

where:

* (I_{\mathrm{render}}): neutral or randomized RGB rendering,
* (D_{\mathrm{render}}): expected depth,
* (M_{\mathrm{render}}): object silhouette,
* (N_{\mathrm{render}}): surface normals.

Render-and-compare approaches are established for evaluating and refining 6D object-pose hypotheses.

---

## 5. RGB pose score

Crop the observed RGB image around the projected candidate and extract spatial DINO features:

[
F_o=f_{\mathrm{DINO}}(I_{\mathrm{observed}})
]

Extract corresponding features from the rendered candidate:

[
F_r=f_{\mathrm{DINO}}(I_{\mathrm{render}})
]

Construct comparison features:

[
F_c =
[
F_o,,
F_r,,
|F_o-F_r|,,
F_o\odot F_r
]
]

A small MLP or shallow convolutional comparison head then predicts:

[
s_{\mathrm{RGB}}(T_i)
=====================

h_{\mathrm{pose}}(F_c)
]

Using spatial patch features is preferable to comparing only global `[CLS]` vectors because orientation differences often appear locally at corners, openings, markings and visible side faces.

The head may optionally predict a correction:

[
(\Delta x,\Delta y,\Delta z,\Delta\psi)
]

but candidate ranking should be implemented first.

---

## 6. Depth-based pose score

Compare the rendered candidate depth with the observed projected depth.

For pixels where both values are valid:

[
r_D(u,v)
========

## D_{\mathrm{observed}}(u,v)

D_{\mathrm{render}}(u,v)
]

Useful scores include:

### Surface support

[
s_{\mathrm{support}}
====================

\frac{
#{|r_D|<\epsilon}
}{
#{\text{testable rendered pixels}}
}
]

### Free-space violation

A strong contradiction exists when the rendered object predicts a surface closer to the camera than the observed scene:

[
D_{\mathrm{render}}
<
D_{\mathrm{observed}}-\epsilon
]

This means that the hypothesis predicts object material in an apparently empty region.

### Occlusion handling

When:

[
D_{\mathrm{observed}}
<
D_{\mathrm{render}}-\epsilon
]

another surface may be occluding the candidate. Such pixels should generally be ignored or weakly penalized rather than treated as proof that the candidate is wrong.

Missing depth should also be treated as unknown, not automatically as background.

### Additional scores

The validator can also use:

[
s_{\mathrm{normal}}
===================

\operatorname{mean}
\left(
N_{\mathrm{observed}}^\top
N_{\mathrm{render}}
\right)
]

and, when an observed object mask is available:

[
s_{\mathrm{silhouette}}
=======================

\operatorname{IoU}
\left(
M_{\mathrm{observed}},
M_{\mathrm{render}}
\right)
]

---

## 7. Final candidate score

Initially, keep the scoring modules interpretable:

[
S(T_i)
======

w_gS_{\mathrm{geometry}}
+
w_rS_{\mathrm{RGB}}
+
w_dS_{\mathrm{depth}}
+
w_sS_{\mathrm{silhouette}}
+
w_nS_{\mathrm{normal}}
]

Select:

[
T^*=\arg\max_{T_i}S(T_i)
]

Later, the individual scores and residual statistics can be passed to a small learned fusion MLP.

For symmetric concrete blocks, some orientations may be fundamentally indistinguishable. In that case, the result should be represented as an equivalence class rather than forcing an arbitrary orientation:

[
T
\sim
T R_z(90^\circ)
]

RGB can resolve the ambiguity only when texture, damage, markings, holes or visible structural differences provide orientation information.

---

## 8. Training strategy

### RGB detection head

Train using:

* manually annotated real RGB images,
* synthetic renderings of the block models,
* gripper occlusions,
* variable lighting and shadows,
* concrete texture variation,
* dust and background clutter.

Start with the DINO backbone frozen. Unfreeze only the last transformer blocks if the dense head does not adapt sufficiently.

### Pose-validation head

Generate training pairs from known poses.

Positive examples:

[
|\Delta t|<\epsilon_t,
\qquad
|\Delta\psi|<\epsilon_\psi
]

Hard negative examples should particularly include:

* (90^\circ), (180^\circ) and (270^\circ) yaw offsets,
* small translation errors,
* slightly incorrect height,
* wrong block type,
* incorrect placement under partial occlusion.

A ranking loss is especially appropriate:

[
S(T_{\mathrm{correct}})

>

S(T_{\mathrm{incorrect}})+m
]

because the practical task is to select the best candidate rather than merely classify candidates independently.

---

## 9. Recommended implementation sequence

1. Retain the existing geometric PCL detector as the baseline.
2. Add RGB-DINO block segmentation with a simple patch-level head.
3. Back-project RGB detections and measure recovered failure cases.
4. Add deterministic rendered-depth candidate validation.
5. Add DINO-based RGB candidate comparison.
6. Add the depth-image block detector only when experiments show that it recovers additional failures beyond the RGB and PCL branches.
7. Replace the patch-level RGB head with a DPT-style decoder only when higher mask resolution is necessary.

The recommended final architecture is therefore:

[
\boxed{
\begin{aligned}
&\text{Primary: geometric PCL detector}\
&\text{Recovery: DINO RGB segmentation + depth-image segmentation}\
&\text{Validation: RGB feature comparison + rendered-depth consistency}\
&\text{Output: geometrically valid, multimodally ranked pose}
\end{aligned}
}
]
