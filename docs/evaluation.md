# Evaluación robot–humano: scores y métricas

Este documento describe cómo se calculan todas las métricas y scores en `test/retargeting/robot2human.py` para evaluar la fidelidad de imitación entre movimiento humano (`.npz`) y movimiento retargeteado del robot (`.pkl`).

---

## Intuición: ¿qué mide cada cosa?

En pocas palabras: **comparas el movimiento del humano con el del robot** después de “ponerlos en el mismo mundo” (misma escala, misma orientación, centrados en la pelvis). Luego mides **qué tan bien coinciden las poses y el ritmo** (imitación) y **qué tan “sano” es el movimiento del robot** (calidad). Abajo va la intuición de cada paso y cada número.

### Preprocesamiento (paso a paso)

1. **Datos de entrada**  
   Del humano tienes posiciones 3D de articulaciones en cada frame; del robot, la misma idea pero reconstruida con MuJoCo a partir de las articulaciones del robot. Solo se usan articulaciones que existen y son válidas en ambos.

2. **Centrado en pelvis**  
   Restas la posición de la pelvis en cada frame a todas las articulaciones. Así dejas de comparar “dónde está el humano en el mundo” vs “dónde está el robot” y te enfocas en **la forma del cuerpo relativa a la pelvis**. Es como poner a ambos con la pelvis en el origen.

3. **Procrustes (escala + rotación)**  
   El robot puede ser más grande o más pequeño y estar rotado respecto al humano. Procrustes encuentra **un factor de escala y una rotación** que, aplicados al robot, hacen que su “silueta” encaje lo mejor posible con la del humano (en sentido de mínimos cuadrados). Después de esto, las métricas de imitación ya no se ven afectadas por tamaño ni orientación global; solo por **qué tan bien coincide la pose**.

---

### Métricas de imitación (¿qué tan bien imita el robot?)

| Métrica | Intuición en una frase |
|--------|-------------------------|
| **MPJPE** | **Error medio de posición:** en promedio, ¿a cuántos centímetros está cada articulación del robot de donde debería estar (según el humano)? Un número bajo = las articulaciones coinciden bien. |
| **PCK@10 cm** | **Porcentaje de “aciertos”:** ¿qué fracción de articulaciones en cada frame está a menos de 10 cm del objetivo? Es como un “porcentaje de keypoints correctos”. Alto = muchas articulaciones muy cerca de la referencia. |
| **Error de velocidad** | **¿Se mueve al mismo ritmo?** No basta con que en un frame la pose coincida; las articulaciones tienen que **moverse** de forma parecida. Esta métrica mide cuánto difiere la velocidad (cambio de posición entre frames) del robot respecto al humano. Bajo = el ritmo y la dirección del movimiento imitan bien. |
| **RMSE trayectoria del root** | **¿La pelvis sigue el mismo camino?** Compara la trayectoria del centro (pelvis) del humano con la del robot en el tiempo. Bajo = el robot “camina” o se desplaza de forma similar al humano. |
| **DTW pose** | **¿Coincide la secuencia de poses en el tiempo?** Permite que los frames no estén perfectamente alineados (uno puede ir un poco adelantado o atrasado). DTW encuentra el mejor “emparejamiento” en el tiempo y mide el coste de esa alineación. Bajo = la secuencia de poses del robot se parece a la del humano incluso si hay pequeños desfases temporales. |

Resumen: **MPJPE y PCK** = precisión de la pose en cada instante; **velocidad** = dinámica; **root RMSE** = trayectoria del cuerpo en el espacio; **DTW** = coherencia temporal de la secuencia de poses.

---

### Métricas de calidad (¿el movimiento del robot es “realista”?)

Estas **no comparan con el humano**; solo miran el movimiento del robot.

| Métrica | Intuición en una frase |
|--------|-------------------------|
| **Tasa de violación de límites** | **¿Respeta los límites articulares?** Las articulaciones del robot tienen rangos (ej. rodilla de 0° a 120°). Esta métrica es la fracción de valores (en el tiempo) que se salen de esos rangos. 0 = nunca se pasa; 1 = siempre fuera de rango. |
| **Aceleración media de DOFs** | **¿El movimiento es suave?** Mide cuánto “acelera” en promedio la configuración articular. Valores muy altos = sacudidas o cambios bruscos; valores moderados = movimiento fluido. |
| **Foot slip** | **¿Los pies resbalan en el suelo?** Cuando el pie está en contacto con el suelo (stance), no debería moverse mucho en horizontal. Esta métrica mide la velocidad horizontal del pie en esos momentos. Bajo = pies bien apoyados; alto = deslizamiento poco realista. |

---

### Scores compuestos (números finales 0–100)

- **Imitation score**  
  Combina MPJPE, PCK@10, error de velocidad y RMSE del root (y opcionalmente DTW) en **un solo número**: “qué tan bien imita”. Cada métrica se convierte en un “sub-score” (con una exponencial para que error bajo → score alto) y se promedian con pesos. **Alto = imitación fiel.**

- **Quality score**  
  Combina violación de límites, foot slip y aceleración en **un solo número**: “qué tan sano/realista es el movimiento del robot”. Empieza en 100 y se multiplica por factores que penalizan violaciones, deslizamiento y sacudidas. **Alto = movimiento físicamente plausible y suave.**

- **Overall score**  
  Es **80% imitation + 20% quality**. Responde: “en conjunto, ¿qué tan bueno es este retargeting?” Prioriza que imite bien, pero también castiga movimientos imposibles o muy bruscos.

---

### Resumen en una línea

- **Imitación:** “¿El robot hace la misma figura y se mueve como el humano?”  
- **Calidad:** “¿Ese movimiento podría hacerlo un robot real sin romperse o patinar?”  
- **Overall:** “Imitación buena (80%) + calidad aceptable (20%).”

---

## 1. Preprocesamiento y alineación

### 1.1 Datos de entrada

- **Humano:** posiciones 3D de articulaciones por frame, $\mathbf{P}^{\text{h}} \in \mathbb{R}^{T \times J \times 3}$, con $T$ frames y $J$ articulaciones (orden fijo: pelvis, caderas, rodillas, etc.).
- **Robot:** se reconstruye la pose en cada frame con MuJoCo (`qpos` → `mj_forward`) y se extraen las posiciones 3D de los cuerpos equivalentes, $\mathbf{P}^{\text{r}} \in \mathbb{R}^{T \times J \times 3}$.

Solo se usan articulaciones **válidas**: aquellas con datos finitos en humano y robot en todos los frames. Sean $\mathcal{J}$ el conjunto de índices válidos y $\mathbf{P}^{\text{h}}_{\text{val}}, \mathbf{P}^{\text{r}}_{\text{val}} \in \mathbb{R}^{T \times |\mathcal{J}| \times 3}$.

### 1.2 Centrado en pelvis

Se centra todo en la pelvis (índice 0) para eliminar traslación global:

$$
\mathbf{C}^{\text{h}}_{t,j} = \mathbf{P}^{\text{h}}_{\text{val},t,j} - \mathbf{P}^{\text{h}}_{\text{val},t,0},
\quad
\mathbf{C}^{\text{r}}_{t,j} = \mathbf{P}^{\text{r}}_{\text{val},t,j} - \mathbf{P}^{\text{r}}_{\text{val},t,0}.
$$

### 1.3 Ajuste Procrustes (escala + rotación)

Se busca un escalar $s > 0$ y una rotación $\mathbf{R} \in \mathrm{SO}(3)$ que alineen el robot al humano.

**Escala:** se elige $s$ que minimiza (en mínimos cuadrados) $\| s \,\mathbf{C}^{\text{r}}_{\text{flat}} - \mathbf{C}^{\text{h}}_{\text{flat}} \|^2$, con las matrices aplanadas en filas. La solución es:

$$
s = \frac{\sum (\mathbf{C}^{\text{r}} \odot \mathbf{C}^{\text{h}})}{\sum (\mathbf{C}^{\text{r}} \odot \mathbf{C}^{\text{r}})} \,, \qquad
s \leftarrow \max(s, 10^{-6}).
$$

En código: `scale = sum(src * tgt) / sum(src * src)`.

**Rotación (Orthogonal Procrustes):** con $\mathbf{S} = s \,\mathbf{C}^{\text{r}}_{\text{flat}}$ se define $\mathbf{H} = \mathbf{S}^\top \mathbf{C}^{\text{h}}_{\text{flat}}$. Si $\mathbf{H} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$ (SVD), entonces:

$$
\mathbf{R} = \mathbf{U} \mathbf{V}^\top.
$$

Si $\det(\mathbf{R}) < 0$, se corrige reflección invirtiendo la última columna de $\mathbf{U}$ y se recalcula $\mathbf{R}$.

**Poses alineadas del robot:**

$$
\mathbf{P}^{\text{r}}_{\text{align},t,j} = s \,\mathbf{R} \,\mathbf{C}^{\text{r}}_{t,j}
\quad \Leftrightarrow \quad
\text{(por frame)} \quad \mathbf{P}^{\text{r}}_{\text{align}} = s \,\mathbf{C}^{\text{r}} \mathbf{R}^\top.
$$

En el código, $\mathbf{C}^{\text{h}}$ se denota como `human_center` y la pose alineada del robot como `robot_aligned`.

---

## 2. Métricas de imitación (por frame, centradas)

Todas estas métricas se calculan después del centrado y Procrustes, sobre poses centradas en pelvis y alineadas.

### 2.1 MPJPE (Mean Per Joint Position Error)

Error de posición promedio por articulación y por frame, en **centímetros**:

$$
e_{t,j} = \big\| \mathbf{C}^{\text{h}}_{t,j} - \mathbf{P}^{\text{r}}_{\text{align},t,j} \big\|_2
\quad \Rightarrow \quad
\text{MPJPE}_{\text{cm}} = \frac{100}{T \cdot |\mathcal{J}|} \sum_{t,j} e_{t,j}.
$$

Es decir, $\text{MPJPE}_{\text{cm}} = 100 \cdot \overline{e}$ con $\overline{e}$ la media de las normas en metros.

### 2.2 PCK@10 cm (Percentage of Correct Keypoints)

Porcentaje de pares (frame, articulación) con error por debajo de 10 cm:

$$
\text{PCK@10}_{\%} = \frac{100}{T \cdot |\mathcal{J}|} \,
\big| \big\{ (t,j) : e_{t,j} < 0.10 \big\} \big|.
$$

En código: `pck_10cm = mean(per_joint_err_m < 0.10) * 100`.

### 2.3 Error de velocidad (velocity error)

Velocidades en el espacio centrado (humano y robot alineado):

$$
\mathbf{v}^{\text{h}}_t = \frac{\mathbf{C}^{\text{h}}_{t+1} - \mathbf{C}^{\text{h}}_t}{\Delta t},
\quad
\mathbf{v}^{\text{r}}_t = \frac{\mathbf{P}^{\text{r}}_{\text{align},t+1} - \mathbf{P}^{\text{r}}_{\text{align},t}}{\Delta t},
\quad
\Delta t = \frac{1}{\text{fps}}.
$$

Error de velocidad en **cm/s** (promedio sobre frames y articulaciones):

$$
\text{vel\_error}_{\text{cm/s}} = \frac{100}{ (T-1) \cdot |\mathcal{J}| }
\sum_{t=0}^{T-2} \sum_{j} \big\| \mathbf{v}^{\text{h}}_{t,j} - \mathbf{v}^{\text{r}}_{t,j} \big\|_2.
$$

### 2.4 RMSE de la trayectoria del root (pelvis)

Trayectorias del root relativas al primer frame (sin Procrustes para el root, pero sí misma escala/rotación):

$$
\mathbf{r}^{\text{h}}_t = \mathbf{P}^{\text{h}}_{\text{val},t,0} - \mathbf{P}^{\text{h}}_{\text{val},0,0},
\quad
\mathbf{r}^{\text{r}}_t = \mathbf{P}^{\text{r}}_{\text{val},t,0} - \mathbf{P}^{\text{r}}_{\text{val},0,0},
\quad
\mathbf{r}^{\text{r}}_{\text{align},t} = s \, \mathbf{R} \, \mathbf{r}^{\text{r}}_t.
$$

RMSE en **cm**:

$$
\text{root\_traj\_rmse}_{\text{cm}} = 100 \cdot \sqrt{ \frac{1}{T} \sum_{t=0}^{T-1} \big\| \mathbf{r}^{\text{h}}_t - \mathbf{r}^{\text{r}}_{\text{align},t} \big\|_2^2 }.
$$

### 2.5 DTW pose (Dynamic Time Warping)

Opcional (`--enable-dtw`). Se submuestrea a un máximo de `max_dtw_frames` frames en humano y robot para limitar coste. Cada pose se representa como vector plano de tamaño $3|\mathcal{J}|$.

- $\mathbf{h}_i \in \mathbb{R}^{3|\mathcal{J}|}$: pose humano en “frame” $i$ (submuestreado).
- $\mathbf{r}_j \in \mathbb{R}^{3|\mathcal{J}|}$: pose robot en “frame” $j$ (submuestreado).

Coste local (normalizado por número de articulaciones):

$$
c(i,j) = \frac{\| \mathbf{h}_i - \mathbf{r}_j \|_2}{\sqrt{|\mathcal{J}|}}.
$$

Se usa DTW estándar (ventana no restringida) con este coste. Si $n$ y $m$ son las longitudes de las secuencias submuestreadas y $\text{DTW}(\mathbf{h}, \mathbf{r})$ es el coste total del camino óptimo:

$$
\text{dtw\_pose}_{\text{m}} = \frac{\text{DTW}(\mathbf{h}, \mathbf{r})}{n + m}.
$$

Recurrencia del DP: $D(i,j) = c(i,j) + \min\big( D(i-1,j), D(i,j-1), D(i-1,j-1) \big)$ con $D(0,0)=0$.

En **cm**: $\text{dtw\_pose}_{\text{cm}} = 100 \cdot \text{dtw\_pose}_{\text{m}}$.

---

## 3. Métricas de calidad (solo robot)

### 3.1 Tasa de violación de límites articulares

Para cada articulación con límites (no FREE/BALL), se cuenta cuántos valores de `qpos` en el tiempo salen del rango $[\text{lo}, \text{hi}]$:

$$
\text{joint\_limit\_violation\_rate} = \frac{\# \text{valores con } q \notin [\text{lo}, \text{hi}]}{\# \text{total de valores } q \text{ considerados}}.
$$

Valor en $[0, 1]$; 0 = sin violaciones.

### 3.2 Aceleración media de DOFs

Con $\mathbf{q}_t \in \mathbb{R}^{n_{\text{dof}}}$ las posiciones articulares (sin root) en el frame $t$ y $\Delta t = 1/\text{fps}$:

$$
\mathbf{a}_t = \frac{\mathbf{q}_{t+2} - 2\mathbf{q}_{t+1} + \mathbf{q}_t}{(\Delta t)^2}.
$$

$$
\text{dof\_acc\_mean} = \frac{1}{T-2} \sum_{t=0}^{T-3} \| \mathbf{a}_t \|_2.
$$

Refleja suavidad de la trayectoria en el espacio de articulaciones.

### 3.3 Deslizamiento de pies (foot slip)

Para cada pie (left_foot, right_foot) en coordenadas mundo del robot:

- Velocidad horizontal (xy) entre frames consecutivos:
  $$
  \mathbf{v}^{\text{xy}}_{t} = \frac{\big( \mathbf{p}_{t+1}^{x,y} - \mathbf{p}_{t}^{x,y} \big)}{\Delta t}.
  $$
- Umbral de suelo: $\text{ground} = \text{percentil}_{5\%}(\mathbf{p}^{z})$ (altura del pie).
- **Stance:** frames donde el pie está “en suelo”: $p^z_t \le \text{ground} + 0.03$ (3 cm).

Se promedia la norma de $\mathbf{v}^{\text{xy}}$ solo en frames de stance, y se convierte a **cm/s** (×100). El score final es la media entre ambos pies:

$$
\text{foot\_slip}_{\text{cm/s}} = \frac{100}{2} \sum_{\text{pie}} \frac{1}{|\mathcal{T}_{\text{stance}}^{\text{pie}}|} \sum_{t \in \mathcal{T}_{\text{stance}}^{\text{pie}}} \| \mathbf{v}^{\text{xy}}_t \|_2.
$$

---

## 4. Función auxiliar: score exponencial

Se usa para convertir métricas “a menor mejor” en scores “a mayor mejor” en $[0,1]$:

$$
\text{safe\_exp\_score}(x, \sigma) = \begin{cases}
0 & \text{si } x \notin \mathbb{R} \text{ (nan/inf)}, \\
\exp\big( - \frac{x}{\sigma} \big) & \text{en otro caso}.
\end{cases}
$$

Con $\sigma > 0$ fijo por métrica. Así, $x=0 \Rightarrow 1$, y $x \gg \sigma \Rightarrow \approx 0$.

---

## 5. Scores compuestos (rewards)

### 5.1 Imitation score (0–100)

Combinación ponderada de métricas de imitación, mapeadas con $\text{safe\_exp\_score}$ o normalizadas:

$$
\begin{aligned}
S_{\text{base}} &= 100 \cdot \Big(
  0.45 \cdot \text{safe\_exp\_score}(\text{MPJPE}_{\text{cm}}, 20) \\
&\quad + 0.20 \cdot \frac{\text{PCK@10}_{\%}}{100} \\
&\quad + 0.20 \cdot \text{safe\_exp\_score}(\text{vel\_error}_{\text{cm/s}}, 80) \\
&\quad + 0.15 \cdot \text{safe\_exp\_score}(\text{root\_traj\_rmse}_{\text{cm}}, 30)
\Big).
\end{aligned}
$$

Si DTW está habilitado y $\text{dtw\_pose}_{\text{cm}}$ es finito:

$$
S_{\text{imitation}} = 0.85 \cdot S_{\text{base}} + 15 \cdot \text{safe\_exp\_score}(\text{dtw\_pose}_{\text{cm}}, 25).
$$

Si no: $S_{\text{imitation}} = S_{\text{base}}$. Finalmente se recorta:

$$
S_{\text{imitation}} \leftarrow \text{clip}(S_{\text{imitation}}, 0, 100).
$$

Pesos: 45% MPJPE, 20% PCK@10, 20% error de velocidad, 15% RMSE root; y si hay DTW, 85% de lo anterior + 15% del término DTW (escala 25 cm).

### 5.2 Quality score (0–100)

Se parte de 100 y se multiplica por factores que penalizan violaciones, deslizamiento y aceleraciones altas:

$$
S_{\text{quality}} = 100 \cdot
\exp(-80 \cdot \text{joint\_limit\_violation\_rate})
\cdot \text{safe\_exp\_score}(\text{foot\_slip}_{\text{cm/s}}, 30)
\cdot \text{safe\_exp\_score}(\text{dof\_acc\_mean}, 300).
$$

Cualquier término no finito se trata como “sin penalización” (no se multiplica o se usa 1 según implementación). Luego:

$$
S_{\text{quality}} \leftarrow \text{clip}(S_{\text{quality}}, 0, 100).
$$

### 5.3 Overall score (0–100)

Combinación lineal de imitación y calidad:

$$
S_{\text{overall}} = \text{clip}\big( 0.80 \cdot S_{\text{imitation}} + 0.20 \cdot S_{\text{quality}}, \, 0, \, 100 \big).
$$

Resumen de pesos: 80% imitación, 20% calidad.

---

## 6. Resumen de escalas y pesos

| Métrica / score | Unidad / rango | Escala $\sigma$ en exp | Peso (imitation) |
|-----------------|----------------|------------------------|-------------------|
| MPJPE           | cm             | 20                     | 45%               |
| PCK@10          | %              | — (lineal 0–1)         | 20%               |
| vel_error       | cm/s           | 80                     | 20%               |
| root_rmse       | cm             | 30                     | 15%               |
| dtw_pose (opt.) | cm             | 25                     | 15 pts (85% base + 15·exp) |
| joint_limit     | [0,1]          | 80 (en exp)            | quality           |
| foot_slip       | cm/s           | 30                     | quality           |
| dof_acc_mean    | —              | 300                    | quality           |

Todos los scores finales están en $[0, 100]$ y **a mayor valor, mejor** (rewards).
