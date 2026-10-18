# Tres filtros dejan veintitrés supervivientes

De unas 122 entradas únicas revisadas en adaptación de dominio, aprendizaje por transferencia, destilación, adaptación sin fuente (SFDA) y adaptación en tiempo de prueba (TTA), solo **23 superan los tres filtros**: alineación condicional por clase (F1), ausencia de etiquetas objetivo (F2) y ponderación o filtrado de pseudo-etiquetas por confianza (F3). De esas 23, **16 tienen el mecanismo verificado en artículo o código**: CDAN+E, SRADA, CAN/CDD, DCAN/CMMD, TPN, RWOT, CREDA, PFAN, ProDA, ATDOC-NA, SHOT++, DaC, CoWA-JMDS, A2Net, PLUE y T3A. Ninguna usa dos codificadores con el de origen congelado junto con un término condicional, y ninguna alinea **operadores de covarianza** por clase: todas comparan medias, prototipos o embeddings de media (primer orden), o usan la covarianza solo para puntuar muestras (CoWA). La confianza en sí no es una contribución, porque la versión publicada de CREDA ya introdujo el peso $w=1-\hat H_2(g)/\ln C$, el enrutamiento duro por argmax y la forma de $J_c$. Lo que la formulación de CREDA con dos codificadores aporta frente a esa versión es otra cosa: la mezcla equilibrada por congruencia que vuelve exacta la identidad $J_c=\ln\!\big(\tfrac12\cosh\frac{H_2^t-H_2^s}{2}+\tfrac12e^{-D_{CS}}\big)$, la normalización r29 (con $\ell_c=0\Leftrightarrow D_{CS}=0$), la compuerta $q_c$ y el codificador de origen congelado. A cambio, carece de la repulsión interclase de CDD y DARES, del enrutamiento blando de LMMD, de pesos desacoplados (todos los supervivientes revisados en código los desacoplan), de umbrales adaptativos y del suelo anticolapso de DARES. Quedan tres defectos abiertos: $w$ no está desacoplado, $r$ conserva el gradiente en el numerador de $\ell_c$ y $\alpha=2$ no tiene optimalidad publicada. Las matemáticas no permiten afirmar que CREDA gane. Sí permiten afirmar que ocupa una celda vacía y que algunos de sus rasgos tienen consecuencias formales demostrables. Que esos rasgos ganen solo puede decidirlo una ablación.

**Convención de evidencia.** Las etiquetas proceden de las notas de investigación del 2026-10-10. **[verificado]**: título, autores, venue o mecanismo confirmados en una fuente abierta durante la investigación. **[code]**: leído en el repositorio oficial. **[paper]**: ecuación leída en el texto del artículo. **[memory]**: ecuación reconstruida de memoria por el investigador; debe contrastarse con el PDF antes de citarla en la tesis. **[UNVERIFIED]**: sin fuente que la respalde. **[derivado]**: álgebra propia; las identidades centrales sobre $J_c$ se comprobaron numéricamente en las notas (por ejemplo, $-0{,}421372$ por ambos lados con un kernel gaussiano aleatorio). **Toda reescritura de un método en la notación de CREDA es [derivado] por definición**, porque traslada un método publicado a un escenario que su artículo no contempla.

**Lectura del embudo.** Se usa la lectura **estricta**. F1 exige estructura condicional explícita por clase, y una selección equilibrada por clase (CBST) no cuenta. F3 exige un mecanismo explícito: umbral, peso entrópico, puntuación de fiabilidad o filtro de muestras o de clases. Los métodos en los que la predicción blanda $p(c\mid x)$ es a la vez enrutamiento y peso (LMMD, CPDA, PCT) se clasifican como **supervivientes blandos**, en una subcategoría propia. Cuando un método se descarta en F1 pero existe una variante condicional publicada, la variante vuelve a entrar al embudo (por ejemplo, DANN→CDAN→CDAN+E o SHOT→SHOT++).

---

# PARTE A. Resumen matemático para entender cada método

Esta parte está escrita para la estudiante. Cada método aparece en **su propia notación original**, con una o dos ecuaciones y una explicación breve: qué mide, cómo usa las etiquetas o pseudo-etiquetas y por qué cae en su filtro o sobrevive. La traducción a la notación de CREDA está en la Parte B.

Notación común de esta parte, salvo que el método indique otra: $G$ o $F$ es el extractor de características, $C$ el clasificador, $D$ un discriminador de dominio, $\hat y=\mathrm{softmax}(C(G(x)))$ la predicción, $H(\cdot)$ la entropía de Shannon, $\phi$ el mapa de un kernel $k$ y GRL la capa de inversión de gradiente.

## A.1 Familia adversaria: un crítico aprendido que juega contra el extractor

**DANN / RevGrad (ICML 2015). Cae en F1.**
$$\min_{G,C}\max_D\ \mathcal L_{cls}(C\circ G;S)-\lambda\big[\mathbb E_s\log D(G(x))+\mathbb E_t\log(1-D(G(x)))\big]\quad\text{[memory]}$$
Un discriminador intenta distinguir origen de objetivo. El extractor, a través de la GRL, aprende características que lo confundan. Solo alinea la distribución marginal $P(Z)$ y nunca mira la clase, por eso cae en F1. Sus variantes condicionales, CDAN y MADA, vuelven a entrar al embudo ([TLlib](https://github.com/thuml/Transfer-Learning-Library)) [verificado].

**ADDA (CVPR 2017). Cae en F1. Es el hermano arquitectónico de CREDA con dos codificadores.**
$$\text{Fase 2: }M_t\leftarrow M_s;\ \ \min_D -\mathbb E_s\log D(M_s(x_s))-\mathbb E_t\log(1-D(M_t(x_t))),\qquad \min_{M_t}-\mathbb E_t\log D(M_t(x_t))\quad\text{[memory]}$$
Primero se entrenan un codificador $M_s$ y un clasificador sobre el origen. Después se copia el codificador para el objetivo, se **congelan $M_s$ y $C$** y solo se mueve $M_t$, hasta que un discriminador no distingue sus características de las del origen fijo. Tiene la misma topología que CREDA con dos codificadores: el de origen congelado y el objetivo inicializado desde el origen. Pero el discriminador no ve la clase, así que cae en F1. No se encontró ninguna variante condicional publicada de ADDA ([CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.html)) [verificado].

**CoGAN, PixelDA y CyCADA (2016–2018). Caen en F1.** Las tres trabajan en el espacio de píxeles. CoGAN comparte capas entre dos GAN. PixelDA traduce imágenes de origen al estilo objetivo. CyCADA añade consistencia cíclica y una pérdida semántica $\mathcal L_{sem}=\mathrm{CE}\big(f_s(G_{S\to T}(x_s)),\arg\max f_s(x_s)\big)$ [memory] que preserva la etiqueta durante la traducción pero no alinea $P(Z\mid Y)$. No se trasladan a instancias embebidas ([NeurIPS CoGAN](https://proceedings.neurips.cc/paper/2016/hash/502e4a16930e414107ee22b6198c578f-Abstract.html); [CVF PixelDA](https://openaccess.thecvf.com/content_cvpr_2017/html/Bousmalis_Unsupervised_Pixel-Level_Domain_CVPR_2017_paper.html)) [verificado].

**MADA (AAAI 2018). Pasa F1 y F2, cae en F3.**
$$\mathcal L_{adv}=\sum_{k=1}^{K}\mathbb E_{x\in S\cup T}\,\mathcal L_d\big(D_k(\hat y_k(x)\,G(x)),d_x\big)\quad\text{[memory]}$$
Hay un discriminador por clase, y cada muestra le llega a $D_k$ escalada por su probabilidad de ser de la clase $k$. Es condicional con **enrutamiento blando**. Pero $\hat y_k$ es pertenencia, no fiabilidad: una predicción uniforme sigue repartiendo masa entre los $K$ discriminadores. Por eso cae en F3 ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/11767)) [verificado].

**CDAN (NeurIPS 2018). Cae en F3. Su variante CDAN+E sobrevive.**
$$\min_G\max_D\ -\mathbb E_s\log D(f\otimes g)-\mathbb E_t\log(1-D(f\otimes g)),\qquad g=\hat y\quad\text{[memory]}$$
El discriminador ve el producto exterior entre características y predicciones, así que debe engañarse clase a clase. Usa la predicción entera como condicionamiento blando, sin peso de fiabilidad.

**CDAN+E (NeurIPS 2018). Superviviente.**
$$w(x)=1+e^{-H(g)}\in\big[1+\tfrac1C,\,2\big],\ \text{normalizado dentro de cada dominio}\quad\text{[code]}$$
Las predicciones seguras pesan más en la pérdida del discriminador. Es un énfasis suave y **no una compuerta**: una muestra totalmente incierta conserva peso $1+1/C$. En el código oficial, la entropía lleva un *hook* de GRL (`entropy.register_hook(grl_hook(coeff))`), así que el peso se entrena de forma adversaria. La reimplementación de TLlib, en cambio, desacopla la predicción y el peso ([thuml/CDAN loss.py](https://raw.githubusercontent.com/thuml/CDAN/master/pytorch/loss.py); [TLlib cdan.py](https://raw.githubusercontent.com/thuml/Transfer-Learning-Library/master/tllib/alignment/cdan.py)) [code].

**DAAN (ICDM 2019). Cae en F3.**
$$\mathcal L=\mathcal L_{cls}-\lambda\big[(1-\omega)\mathcal L_g+\omega\textstyle\sum_c\mathcal L_l^c\big]\quad\text{[memory]}$$
Combina MADA con un discriminador global, y un factor dinámico $\omega$ (estimado por distancias $\mathcal A$) decide cuánto pesa lo marginal frente a lo condicional. $\omega$ equilibra términos globales y no pondera muestras por confianza ([arXiv 1909.08184](https://arxiv.org/pdf/1909.08184)) [verificado].

**MCD (CVPR 2018) y SWD (CVPR 2019). Caen en F3.**
$$\text{MCD: }\min_{C_1,C_2}\mathcal L_s-\mathbb E_t|p_1-p_2|_1,\quad \min_G\mathbb E_t|p_1-p_2|_1;\qquad \text{SWD: }\tfrac1M\textstyle\sum_m W_1(\langle\theta_m,p_1\rangle,\langle\theta_m,p_2\rangle)\quad\text{[memory]}$$
Dos clasificadores se empujan a discrepar sobre el objetivo, y el extractor a que coincidan, lo que aleja las muestras de las fronteras de decisión. SWD cambia la discrepancia $L_1$ por una Wasserstein en rodajas. La condicionalidad es implícita (vive en el espacio de predicciones) y no hay pesos ([CVF SWD](https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Sliced_Wasserstein_Discrepancy_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.html)) [verificado].

**MDD (ICML 2019). Cae en F3.**
$$\min_{f,\psi}\mathbb E_s\mathcal L(f(\psi(x)),y)+\eta\big[\gamma\,\mathbb E_s\log\sigma_{h_f}(f'(\psi(x)))+\mathbb E_t\log(1-\sigma_{h_f}(f'(\psi(x))))\big],\ \ h_f=\arg\max f\quad\text{[memory]}$$
Una copia adversaria del clasificador intenta discrepar de las etiquetas predichas (duras) más en el objetivo que en el origen, con un margen $\gamma$. Es el adversario clásico más fuerte en TLlib (89,6 / 69,7 / 79,8 en Office-31 / Office-Home / VisDA), pero no pondera por confianza ([TLlib README](https://raw.githubusercontent.com/thuml/Transfer-Learning-Library/master/examples/domain_adaptation/image_classification/README.md)) [verificado].

**DALN (CVPR 2022). Cae en F3.**
$$\mathrm{NWD}=\tfrac1{N_s}\|C(G(X_s))\|_*-\tfrac1{N_t}\|C(G(X_t))\|_*\quad\text{[memory]}$$
El clasificador se reutiliza como crítico, y la diferencia entre dominios se mide con la norma nuclear de la matriz de predicciones del lote. No necesita discriminador, pero sigue siendo un juego min–max sin pesos explícitos ([CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Reusing_the_Task-Specific_Classifier_as_a_Discriminator_Discriminator-Free_Adversarial_Domain_CVPR_2022_paper.html)) [verificado].

**f-DAL (ICML 2021). Cae en F3.** $D_f(S\|T)\ge\sup_{h'}\mathbb E_s[\ell(h',h)]-\mathbb E_t[\phi^*(\ell(h',h))]$ [memory]. Generaliza DANN a cualquier $f$-divergencia, con un crítico en el espacio de hipótesis. Es condicional solo de forma implícita y no tiene pesos ([NVIDIA](https://research.nvidia.com/labs/toronto-ai/fDAL)) [verificado].

**SymNets (CVPR 2019) y DADA (AAAI 2020). Caen en F3.** SymNets fuerza a que cada muestra sea igualmente probable como «clase $k$ de origen» y «clase $k$ de objetivo», con la pérdida $-\tfrac12\log\sum_kp^{st}_k-\tfrac12\log\sum_kp^{st}_{k+K}$ [memory]. DADA fusiona clasificador y discriminador en una única softmax de $K+1$ salidas, de modo que clase y dominio compiten [UNVERIFIED en la forma exacta]. Ambas son condicionales sin ponderación ([arXiv 1904.04663](https://arxiv.org/abs/1904.04663); [AAAI DADA](https://ojs.aaai.org/index.php/AAAI/article/view/6054)) [verificado].

**CAT (ICCV 2019), CGDM (CVPR 2021) e InAIt (ICCV 2021). Caen en F3 de forma tentativa.** CAT alinea medias de clase $\sum_k\|\mu_k^s-\mu_k^t\|^2$ con pseudo-etiquetas de un *mean teacher*: es un análogo de primer orden del término de CREDA. CGDM exige que las pseudo-etiquetas objetivo generen gradientes del clasificador alineados con los de origen, $1-\cos(\nabla_C\mathcal L_s,\nabla_C\mathcal L_t)$. InAIt estima $p_t(y)$ antes de alinear $p(x\mid y)$ para corregir el desplazamiento de etiquetas. No se leyó el PDF de ninguno, así que no se pudo confirmar si filtran por confianza [UNVERIFIED] ([CVF CAT](https://openaccess.thecvf.com/content_ICCV_2019/html/Deng_Cluster_Alignment_With_a_Teacher_for_Unsupervised_Domain_Adaptation_ICCV_2019_paper.html); [CVF CGDM](https://openaccess.thecvf.com/content/CVPR2021/html/Du_Cross-Domain_Gradient_Discrepancy_Minimization_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.html); [CVF InAIt](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Adversarial_Unsupervised_Domain_Adaptation_With_Conditional_and_Label_Shift_Infer_ICCV_2021_paper.html)) [verificado como venue].

**ToAlign (NeurIPS 2021) y NOUN/PRONOUN (TIP 2021). Caen en F3.** ToAlign alinea solo la parte de cada característica de origen relevante para su clase verdadera, $f^p=w_y\odot f$, así que la condicionalidad existe solo en el lado de origen. NOUN sustituye $\hat y$ por un condicionador normalizado, menos sensible a softmax sobreconfiadas. Ninguno pondera por confianza ([NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/731c83db8d2ff01bdc000083fd3c3740-Paper.pdf); [arXiv 2003.13274](https://arxiv.org/abs/2003.13274v4)) [verificado].

**SRADA (IJCAI 2020). Superviviente con ponderación invertida.**
$$\mathcal L_{adv}=\textstyle\sum_x u(x)\,\ell_d(D(G(x)),d_x),\quad u\ \text{creciente en }H(\hat y)\quad\text{[UNVERIFIED en la forma exacta]}$$
Da **más** fuerza adversaria a las muestras inciertas, mal alineadas, y añade una pérdida *triplet* sobre pseudo-etiquetas. Es justo la dirección contraria a CDAN+E y a CREDA, y ambas son defendibles. Esa tensión es un eje legítimo de ablación ([arXiv 2006.00223](https://arxiv.org/abs/2006.00223)) [verificado: resumen].

**BACG (arXiv 2023, retirado).** Combina dos discriminadores, alineación de gradientes por clase e incertidumbre evidencial de Dirichlet. Pasaría los tres filtros, pero el artículo fue retirado y no debe citarse como precedente revisado ([arXiv 2310.13959](https://arxiv.org/abs/2310.13959)) [verificado].

**Envoltorios adversarios: BSP, GVB, SDAT, ELS y GH/GH++. Caen en F1.** Ninguno es condicional por sí mismo. BSP penaliza los valores singulares dominantes, $\sum_{i\le k}(\sigma_{s,i}^2+\sigma_{t,i}^2)$. GVB añade un puente residual decreciente. SDAT aplica SAM solo a la pérdida de tarea. ELS suaviza las etiquetas de dominio. GH armoniza el ángulo entre los gradientes de alineación y de tarea. Cuando se apilan sobre CDAN+E heredan su estado en el embudo. Importan por otra razón: documentan la inestabilidad del juego adversario, que CREDA evita ([PMLR BSP](https://proceedings.mlr.press/v97/chen19i.html); [ICML SDAT](https://icml.cc/virtual/2022/poster/17769); [ICLR ELS](https://iclr.cc/virtual/2023/poster/11317); [arXiv GH](https://arxiv.org/abs/2408.00288v1)) [verificado].

**RADA (ICCV 2021) y TADA (AAAI 2019). Caen en F1 aunque ponderan por confianza, porque lo hacen sobre la variable equivocada.** Las dos usan la entropía del **discriminador de dominio**, no la del posterior de clase. TADA usa una atención local $1+(1-H(D_r))$ y una global $1+H(D)$ [paper]. Miden transferibilidad, no fiabilidad de clase, y no deben citarse como precedente del peso $w$ ([CVF RADA](https://openaccess.thecvf.com/content/ICCV2021/html/Jin_Re-Energizing_Domain_Discriminator_With_Sample_Relabeling_for_Adversarial_Domain_Adaptation_ICCV_2021_paper.html); [AAAI TADA](https://ojs.aaai.org/index.php/AAAI/article/download/4472/4350)) [verificado].

## A.2 Familia de discrepancia: kernel, momentos, transporte óptimo y divergencias

**TCA (TNN 2011). Cae en F1.** Minimiza la MMD entre proyecciones, del tipo $\mathrm{tr}(KLK)$ [memory]. Solo alinea la marginal $P(X)$; la variante semisupervisada usa solo etiquetas de origen. Su continuación condicional es JDA ([IEEE](https://ieeexplore.ieee.org/document/5640675)) [verificado].

**MIDA (Yan, Kou y Zhang, T-Cyb 2018). Cae en F1.**
$$\max_W\ \mathrm{tr}(KHK_dH)\ \text{(HSIC entre el subespacio y las características de dominio)}\quad\text{[memory]}$$
Busca un subespacio **independiente de la variable de dominio** (dispositivo, fecha de adquisición). Mide dependencia y no la condiciona a la clase, por eso cae en F1. Es el antecedente natural de la segunda idea de novedad de la Parte B ([arXiv 1603.04535](https://www.arxiv.org/pdf/1603.04535)) [verificado].

**DAN / MK-MMD (ICML 2015) y RTN (NIPS 2016). Caen en F1.** DAN suma la MMD multikernel sobre varias capas, $\sum_l\mathrm{MK\text{-}MMD}^2(\mathcal D_s^l,\mathcal D_t^l)$ [memory]. RTN añade un clasificador residual y minimización de entropía. Las dos son marginales. La continuación condicional de DAN es DSAN/LMMD ([NeurIPS RTN](https://papers.nips.cc/paper_files/paper/2016/hash/ac627ab1ccbdb62ec96e702f07f6425b-Abstract.html)) [verificado].

**CORAL y Deep CORAL (2016), CMD (ICLR 2017) y HoMM (AAAI 2020). Caen en F1.**
$$\text{CORAL: }\tfrac1{4d^2}\|C_S-C_T\|_F^2;\qquad \text{HoMM: }\tfrac1{L^p}\Big\|\tfrac1{n_s}\textstyle\sum_i\phi(x_s^i)^{\otimes p}-\tfrac1{n_t}\sum_i\phi(x_t^i)^{\otimes p}\Big\|_F^2\quad\text{[memory / paper]}$$
CORAL alinea covarianzas, es decir, segundo orden, pero **marginales**. CMD alinea momentos centrales y HoMM potencias tensoriales de orden $p$. Son el contraste más directo con CREDA: alinean segundo orden o más, pero sin condicionar a la clase. El término de agrupamiento solo-objetivo de HoMM usa un umbral $\eta$, pero su término entre dominios es marginal. Su variante condicional es COT ([AAAI CORAL](https://ojs.aaai.org/index.php/AAAI/article/view/10306); [arXiv HoMM](https://arxiv.org/pdf/1912.11976)) [verificado].

**SA (Fernando et al., ICCV 2013) y GFK (Gong et al., CVPR 2012). Caen en F1.**
$$\text{SA: }M^*=\arg\min_M\|X_SM-X_T\|_F^2=X_S^\top X_T,\ \ X_a=X_SX_S^\top X_T;\qquad \text{GFK: }\langle z_i^\infty,z_j^\infty\rangle=\int_0^1\big(\Phi(t)^\top x_i\big)^\top\big(\Phi(t)^\top x_j\big)\,dt=x_i^\top Gx_j\quad\text{[paper]}$$
Las dos representan cada dominio con **un solo subespacio** de dimensión $d$, los $d$ vectores propios principales de su covarianza (PCA). SA lleva la base de origen $X_S$ hacia la de objetivo $X_T$ en forma cerrada; GFK integra todos los subespacios de la geodésica $\Phi(t)$ del grassmanniano, con $\Phi(0)=P_S$ y $\Phi(1)=P_T$, y lo resume en el kernel $G$. Alinean estructura de segundo orden **marginal**, como CORAL, y no tienen ningún término por clase, por eso caen en F1. No usan etiquetas del objetivo. Las etiquetas de origen solo entran en el clasificador final o, en las variantes GFK(PLS, PCA) y SA con ITML, en el subespacio de origen; el subespacio objetivo sigue siendo un PCA sin clase, así que tampoco alinean $P(Z\mid Y=c)$. El PDF de ICCV en CVF devolvió 403; SA se leyó en su versión extendida de arXiv, que cita la de ICCV como trabajo previo ([arXiv SA](https://arxiv.org/abs/1409.5241); [PDF GFK](https://www.cs.utexas.edu/~grauman/papers/subspace-cvpr2012.pdf)) [paper].

**Familia JDA (JDA, BDA, VDA, DICD, ARTL, MEDA, DJP-MMD, JGSA). Pasan F1 y F2, caen en F3.**
$$\min_A\sum_{c=0}^{C}\Big\|\tfrac1{n_c}\textstyle\sum_{x\in\mathcal D_S^{(c)}}A^\top x-\tfrac1{m_c}\sum_{x\in\mathcal D_T^{(c)}}A^\top x\Big\|^2\quad\text{[paper, vía Zhuang et al.]}$$
Se compara la media de clase $c$ del origen con la media de los puntos objetivo que el clasificador actual llama $c$; el caso $c=0$ es el término marginal. Las pseudo-etiquetas son **duras** y se refinan iterando: se reentrena el clasificador, se reetiqueta y se repite. **Todas las pseudo-etiquetas pesan lo mismo**, por eso toda la familia cae en F3. Las variantes difieren en un detalle cada una. BDA equilibra marginal y condicional con $\mu$. DICD y VDA añaden dispersión discriminativa. ARTL y MEDA aprenden un clasificador en el RKHS, MEDA con un factor $\mu$ dinámico. DJP-MMD mide directamente la conjunta y resta una discrepancia interclase. JGSA aprende **dos proyecciones**, una por dominio, acopladas por $\|A-B\|_F^2$, y es el análogo superficial más cercano a los dos codificadores de CREDA ([arXiv 1911.02685](https://arxiv.org/pdf/1911.02685); [arXiv BDA](https://arxiv.org/pdf/1807.00516); [arXiv JGSA](https://arxiv.org/pdf/1705.05498); [arXiv MEDA](https://arxiv.org/pdf/1807.07258); [arXiv DJP-MMD](https://arxiv.org/abs/1912.00320)) [verificado].

**JAN / JMMD (ICML 2017). Cae en F3.**
$$\mathrm{JMMD}=\tfrac1{n_s^2}\textstyle\sum\prod_lk^l(z_i^{sl},z_j^{sl})+\tfrac1{n_t^2}\sum\prod_lk^l(z_i^{tl},z_j^{tl})-\tfrac2{n_sn_t}\sum\prod_lk^l(z_i^{sl},z_j^{tl})\quad\text{[paper]}$$
Es una MMD sobre la conjunta de varias capas, y una de ellas es la softmax. Las predicciones blandas entran como un factor del kernel producto, sin peso por muestra ([arXiv 1605.06636](https://arxiv.org/abs/1605.06636)) [verificado].

**WMMD (CVPR 2017). Cae en F3.** $\big\|\frac{1}{\sum\alpha_{y_i}}\sum_i\alpha_{y_i^s}\phi(x_i^s)-\frac1N\sum_j\phi(x_j^t)\big\|^2$, con $\alpha_c=\hat w_c^t/w_c^s$ [paper]. Repondera las clases del origen para que su media tenga la mezcla de clases del objetivo, lo que corrige el desplazamiento de etiquetas. Los pesos son proporciones de clase, no confianzas ([arXiv 1705.00609](https://arxiv.org/pdf/1705.00609)) [verificado].

**CAN / CDD (CVPR 2019). Superviviente.**
$$\hat D^{cdd}=\tfrac1M\textstyle\sum_c\hat D^{cc}-\tfrac1{M(M-1)}\sum_c\sum_{c'\ne c}\hat D^{cc'},\qquad \hat D^{c_1c_2}=e_1+e_2-2e_3\quad\text{[paper]}$$
Acerca entre dominios las nubes de kernel de la misma clase y **aleja las de clases distintas**. Las etiquetas objetivo salen de un k-means esférico inicializado en los centros de clase de origen. Una muestra se conserva solo si $\mathrm{dist}(\phi(x_t),O^{t(\hat y)})<D_0$, y una clase solo si tiene más de $N_0$ muestras retenidas. Sobrevive por filtrado binario, no por un peso continuo ([arXiv 1901.00976](https://arxiv.org/pdf/1901.00976); [código](https://github.com/kgl-prml/Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation)) [paper + code].

**DSAN / LMMD (TNNLS 2021). Superviviente blando.**
$$\hat d=\tfrac1C\textstyle\sum_c\big\|\sum_iw_i^{sc}\phi(x_i^s)-\sum_jw_j^{tc}\phi(x_j^t)\big\|^2,\qquad w_j^{tc}=\hat y_{jc}/\textstyle\sum_k\hat y_{kc}\quad\text{[paper]}$$
Es una MMD por clase en la que **cada muestra objetivo contribuye a todas las clases** en proporción a su probabilidad predicha. En el código oficial los pesos están desacoplados (`.cpu().data.numpy()`), y solo entran las clases presentes a la vez en las etiquetas de origen y en el conjunto argmax objetivo. Sobrevive solo por la cláusula blanda, porque la probabilidad es a la vez enrutamiento y peso ([arXiv 2106.09388](https://arxiv.org/pdf/2106.09388); [lmmd.py](https://raw.githubusercontent.com/jindongwang/transferlearning/master/code/DeepDA/loss_funcs/lmmd.py)) [paper + code].

**DCAN / CMMD (Pattern Recognition, 2022/2023). Superviviente.**
$$\mathcal L_{CMMD}=\mathrm{Tr}(K_s\tilde L_s^{-1}L_s\tilde L_s^{-1})+\mathrm{Tr}(K_t\tilde L_t^{-1}L_t\tilde L_t^{-1})-2\,\mathrm{Tr}(L_{ts}\tilde L_s^{-1}K_{st}\tilde L_t^{-1}),\quad \tilde L=L+\lambda I\quad\text{[paper, reconstrucción parcial]}$$
Compara los **operadores de embedding condicional** $Z\mid Y$ de ambos dominios, es decir, cómo varían las características con la etiqueta. Una muestra objetivo recibe pseudo-etiqueta solo si $\max\hat y>\gamma_0=0{,}95$. Trabaja a nivel de operador, pero el operador es condicional ($Z\mid Y$) y no de covarianza dentro de la clase, y exige invertir una Gram regularizada ([arXiv 2006.07776](https://arxiv.org/pdf/2006.07776)) [paper].

**TPN (CVPR 2019). Superviviente.**
$$L_G=\tfrac1C\textstyle\sum_c\big[\|\tilde\mu_c^s-\tilde\mu_c^t\|^2_{\mathcal H}+\|\tilde\mu_c^s-\tilde\mu_c^{st}\|^2_{\mathcal H}+\|\tilde\mu_c^t-\tilde\mu_c^{st}\|^2_{\mathcal H}\big]\quad\text{[paper, primer término; resto reconstruido]}$$
Usa tres prototipos por clase: solo origen, solo objetivo pseudo-etiquetado y la unión. La pseudo-etiqueta se asigna por el prototipo de origen más cercano **solo si la puntuación supera 0,6**, y se reasigna en cada iteración. El prototipo mixto $\mu^{st}$ es el paralelo más cercano a la matriz mixta $\tilde K^{mix}$ de CREDA ([arXiv 1904.11227](https://arxiv.org/pdf/1904.11227)) [paper].

**CPDA (arXiv 2608.09193, agosto de 2026, preprint). Superviviente blando.** $D^2=\sum_c\pi_t(c)\|\hat\mu_{s,c}-\hat\mu_{t,c}\|^2$, con pesos objetivo blandos **desacoplados** por defecto y un umbral de confianza opcional $\tau=0{,}0$ [paper]. Es una MMD por clase sobre trayectorias que incluye MMD, CORAL y HoMM como casos restringidos. Viene con una cota de riesgo, $R_t(h)\le R_s^{\pi_t}(h)+B\sum_c\pi_t(c)\,\mathrm{MMD}(P_s^c,P_t^c)$ ([arXiv 2608.09193](https://arxiv.org/pdf/2608.09193)) [paper].

**COT (arXiv 2024). Cae en F3.** Es la variante condicional de HoMM: transporte óptimo entre la mezcla de datos y las condicionales de clase de origen, con coste $-\log p_m^S(x)$ y un plan amortizado $T$ [paper]. Las asignaciones son blandas, con términos de entropía, y no hay peso de confianza ([arXiv 2401.15952](https://arxiv.org/pdf/2401.15952)) [verificado].

**JDOT (NIPS 2017) y DeepJDOT (ECCV 2018). Caen en F3.**
$$\min\ \tfrac1{n_s}\textstyle\sum L_s(y_i^s,f(g(x_i^s)))+\sum_{ij}\gamma_{ij}\big[\alpha\|g(x_i^s)-g(x_j^t)\|^2+\lambda_tL_t(y_i^s,f(g(x_j^t)))\big]\quad\text{[paper]}$$
Transporta puntos de origen a puntos objetivo pagando dos cosas: la distancia en características y la pérdida del clasificador en el punto objetivo frente a la etiqueta de origen. La etiqueta entra en el coste, pero sin pesos de fiabilidad ([arXiv 1803.10081](https://arxiv.org/pdf/1803.10081)) [paper].

**ETD (CVPR 2020). Cae en F3 (en el límite).** $C=S\odot C$, con una atención aprendida $S_{ij}$ sobre las predicciones de ambos lados [paper]. Las ponderaciones por pares salen de una similitud aprendida, no de la confianza ([CVF PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Enhanced_Transport_Distance_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf)) [paper].

**RWOT (CVPR 2020). Superviviente.**
$$Q(i,k)\propto d_{\mathcal A}(k)\,D(i,k)+(2-d_{\mathcal A}(k))\,M(i,k),\qquad Z(i,j)=\|G_f(x_i^s)-G_f(x_j^t)\|^2\big(1-Q(j,y_i^s)\big)\quad\text{[paper]}$$
Transportar un par es barato solo si la muestra objetivo es **fiablemente** de la clase de la muestra de origen. La fiabilidad mezcla dos fuentes según la distancia $\mathcal A$ de cada clase: la cercanía a prototipos de origen, en la que se confía al principio, y la salida afilada del clasificador, en la que se confía cuando los dominios ya se solapan. Es fiabilidad continua y explícita ([CVF PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Reliable_Weighted_Optimal_Transport_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf)) [paper].

**PCT (NeurIPS 2021). Superviviente blando.** $\pi_\theta(\mu_k\mid f_j^t)=\frac{p(\mu_k)\exp(\mu_k^\top f_j^t)}{\sum_{k'}p(\mu_{k'})\exp(\mu_{k'}^\top f_j^t)}$ [paper]. Es transporte bidireccional blando entre las características objetivo y los prototipos de clase, que son los pesos del clasificador y se mantienen **fijos** durante la alineación. Estima además las proporciones de clase con una actualización de tipo EM ([arXiv 2110.12024](https://arxiv.org/pdf/2110.12024)) [paper].

**MARS (Machine Learning, Springer). Cae en F3.** Usa una Wasserstein entre marginales reponderadas por proporciones de clase estimadas. Es condicional solo débilmente y no tiene confianza ([arXiv 2006.08161](https://arxiv.org/abs/2006.08161v3)) [verificado].

**CS-DA (Yin et al., UAI 2024). Cae en F3. Es el pariente teórico-informacional más cercano.**
$$\hat D_{CS}=\log\tfrac1{M^2}\textstyle\sum\kappa(z^s,z^s)+\log\tfrac1{N^2}\sum\kappa(z^t,z^t)-2\log\tfrac1{MN}\sum\kappa(z^s,z^t)\quad\text{[paper]}$$
Mide en escala logarítmica cuánto se solapan las dos estimaciones de densidad por kernel en relación con su autosolapamiento. Una versión condicional, $D_{CCS}$, alinea $p(\hat y\mid z)$ usando una Gram $L$ construida con las predicciones **blandas**. El propio artículo advierte que alinea $p(y\mid z)$ y no $p(z\mid y)$. No hay pesos ni umbrales, por eso cae en F3 ([arXiv 2405.19978](https://arxiv.org/pdf/2405.19978)) [paper].

**RJSD (Hoyos-Osorio y Sánchez Giraldo, arXiv 2023). Cae en F1 en la lectura estricta.**
$$D_{JS}^{\mathcal H}(P,Q)=S\big(\tfrac12(C_P+C_Q)\big)-\tfrac12\big[S(C_P)+S(C_Q)\big],\quad S=\text{entropía de von Neumann}\quad\text{[paper]}$$
Es una divergencia de Jensen–Shannon entre operadores de covarianza no centrados. Es nula si y solo si $P=Q$ con $\kappa^2$ característico, y cuesta $O(n^3)$. Su uso en adaptación sustituye la MMD de JAN por RJSD sobre la conjunta de capas, sin enrutamiento por clase. Es la alternativa de principio con $\alpha=1$ al $J_c$ de CREDA ([arXiv 2305.16446](https://arxiv.org/abs/2305.16446)) [paper].

**CREDA (Pérez-Rosero, Álvarez-Meza y Castellanos-Domínguez, *Mathematics* 13(16):2602, 2025). Superviviente y precursor directo.**
$$\tilde I_2(K^s_c;\tilde K^t_c)=\tfrac12\big(H_2(K^s_c)+H_2(\tilde K^t_c)\big)-H_2(K^{mix}_c),\quad K^{mix}_c=\begin{bmatrix}K^s_c&K^{st}_c\\(K^{st}_c)^\top&\tilde K^t_c\end{bmatrix},\quad \tilde K^t_c=K^t_c\odot w_cw_c^\top\quad\text{[paper]}$$
$$w_j=1-\hat H_2(g_j)/\log C,\qquad \hat H_2(g_j)=-\log\textstyle\sum_cg_{j,c}^2\quad\text{[paper]}$$
Por cada clase compara la «pureza» de las Gram de origen y de objetivo con la de su mezcla, usando la entropía cuadrática de Rényi matricial. Las muestras objetivo se enrutan por argmax, y el bloque objetivo se pondera por la confianza $w$. Hay un solo codificador compartido. El bloque cruzado no está ponderado y la mezcla no está equilibrada. No hay compuerta ni normalización, y $\alpha$ está fijado en 2. Sobrevive a los tres filtros. El PDF se leyó completo (Ecs. 9, 13–23, 29 y 31) ([PDF CREDA](https://mdpi-res.com/d_attachment/mathematics/mathematics-13-02602/article_deploy/mathematics-13-02602.pdf)) [paper].

## A.3 Autoentrenamiento, pseudo-etiquetado y adaptación sin fuente

**CBST (ECCV 2018) y CRST (ICCV 2019). Caen en F1 en la lectura estricta.** Hacen autoentrenamiento con umbrales $k_c$ por clase elegidos para conservar una porción $p$ de cada clase, y CRST añade regularizadores de confianza [memory]. Es selección equilibrada por clase, no alineación de estructura condicional. En una lectura laxa pasarían F1 ([arXiv 1810.07911](https://arxiv.org/abs/1810.07911); [arXiv 1908.09822](https://arxiv.org/abs/1908.09822)) [verificado].

**MSTN (ICML 2018). Cae en F3.** $C_k^S\leftarrow\theta C_k^S+(1-\theta)C_k^S(t)$, y $\mathcal L_{SM}=\sum_k\Phi(C_k^S,C_k^T)$ [paper]. Alinea la media móvil de cada clase. Sus autores «dudan de las pseudo-etiquetas» y prefieren promediar el ruido antes que filtrarlo ([PMLR](https://proceedings.mlr.press/v80/xie18c.html)) [paper].

**PFAN (CVPR 2019). Superviviente.**
$$w_j=\mathbb 1[\psi\ge\tau],\quad \psi_k=\cos(G(x_j^t),c_k^S),\quad \tau=\tfrac1{1+e^{-\mu(m+1)}}-0{,}01;\qquad \mathcal L_{apa}=\textstyle\sum_kd(c_k^S,c_k^T)\quad\text{[paper]}$$
Una muestra objetivo recibe pseudo-etiqueta solo si ya está cerca de un prototipo de origen, y el listón $\tau$ baja con un calendario. Después se acercan los prototipos de origen y de objetivo de cada clase, con medias móviles adaptativas ([arXiv 1811.08585](https://arxiv.org/abs/1811.08585)) [paper].

**ProDA (2021, venue sin confirmar). Superviviente (segmentación).** $\hat y=\xi\big(\omega\,p_{t,0}\big)$, con $\omega_k=\mathrm{softmax}_k\big(-(\|f-\eta^{(k)}\|-\min_{k'}\|f-\eta^{(k')}\|)T\big)$, calculado bajo `no_grad` [code]. Una predicción blanda fija se repondera por cercanía a cada prototipo, se convierte en etiqueta dura y se descarta si sigue siendo insegura. Combina la predicción completa con distancias a prototipos ([arXiv 2101.10979](https://arxiv.org/abs/2101.10979); [código](https://github.com/microsoft/ProDA/blob/main/models/adaptation_modelv2.py)) [code].

**ATDOC-NC y ATDOC-NA (CVPR 2021). NC cae en F3; NA sobrevive.**
$$w_j=\max_c\tfrac1K\textstyle\sum_{k\in\mathcal N(j)}m_{k,c},\qquad \mathcal L=\sum_jw_j\,\ell_j\big/\sum_jw_j\quad\text{[code]}$$
NA agrega las predicciones de los $K$ vecinos guardadas en una memoria, y la confianza de ese agregado pondera la CE. El peso y el denominador están **desacoplados** (memoria bajo `no_grad`, denominador con `.item()`). ([código ATDOC](https://raw.githubusercontent.com/tim-learn/ATDOC/main/demo_uda.py)) [code].

**MCC (ECCV 2020) y BNM (CVPR 2020). Caen en F1.** MCC reduce la confusión entre clases con pesos $1+e^{-H}$ desacoplados. BNM maximiza la norma nuclear de la matriz de predicciones. Son regularizadores de predicción sin alineación por clase ([código MCC](https://raw.githubusercontent.com/thuml/Versatile-Domain-Adaptation/master/pytorch/train_image_office.py); [CVF BNM](https://openaccess.thecvf.com/content_CVPR_2020/html/Cui_Towards_Discriminability_and_Diversity_Batch_Nuclear-Norm_Maximization_Under_Label_Insufficient_CVPR_2020_paper.html)) [verificado].

**AdaMatch (ICLR 2022), CST (2021) y PCL (IJCAI 2024).** AdaMatch usa un umbral relativo con alineación **marginal** y cae en F1, aunque pasaría F3. CST combina consistencia cíclica de pseudo-etiquetas con un regularizador de Tsallis y cae en F1. PCL agrupa las características alrededor de los pesos de clase sin ponderar pseudo-etiquetas y cae en F3 ([arXiv 2106.04732](https://arxiv.org/abs/2106.04732); [arXiv 2103.03571](https://arxiv.org/abs/2103.03571); [arXiv 2111.06021](https://arxiv.org/abs/2111.06021)) [verificado].

**SHOT (ICML 2020). Cae en F3.**
$$\mathcal L_{IM}=\mathbb E[H(p)]-H(\mathbb E[p]);\qquad \text{pseudo-etiquetas por centroides bajo `no\_grad`}\quad\text{[code]}$$
Congela el clasificador de origen (la «hipótesis») y entrena un extractor objetivo propio con maximización de información más pseudo-etiquetas por centroides ponderados. El código **calcula** `unknown_weight = 1 - ent/np.log(C)`, la versión Shannon del peso de CREDA, pero **nunca lo usa**: la función solo devuelve las etiquetas. Por eso cae en F3 ([código SHOT](https://raw.githubusercontent.com/tim-learn/SHOT/master/object/image_target.py)) [code].

**SHOT++ (TPAMI 2021). Superviviente.** Dentro de cada clase $k$, las $t_k$ muestras de menor entropía forman el subconjunto «fácil». MixMatch propaga después las etiquetas del subconjunto fácil al difícil [paper]. Las muestras más seguras de cada clase actúan como conjunto etiquetado de un problema semisupervisado ([arXiv 2012.07297](https://arxiv.org/abs/2012.07297)) [paper].

**NRC (NeurIPS 2021), AaD (NeurIPS 2022), G-SFDA (ICCV 2021) y SF(DA)² (ICLR 2024). Caen en F3.** Todos imponen consistencia de vecindario o agrupamiento espectral sobre las características objetivo. La afinidad es estructural (vecino recíproco 1, si no $r=0{,}1$ en NRC) y no expresa confianza ([arXiv 2110.04202](https://arxiv.org/abs/2110.04202); [arXiv 2205.04183](https://arxiv.org/abs/2205.04183); [arXiv 2403.10834](https://arxiv.org/abs/2403.10834)) [verificado].

**DaC (NeurIPS 2022). Superviviente.** `mask = max_prob.ge(0.97)` bajo `no_grad` [code]. Las muestras muy seguras se tratan como «tipo origen» y anclan prototipos de clase. El resto se atrae hacia ellas con contraste y una MMD sobre un banco de memoria ([código DaC](https://github.com/ZyeZhang/DaC/blob/main/VisDA/target.py)) [code].

**A2Net (ICCV 2021). Superviviente a nivel de pares.** $\gamma_{ij}=1$ si $s_{ij}>\mu(t)$, $-1$ si $s_{ij}<l(t)$ y 0 en otro caso, con $s_{ij}=\sigma(p_i)^\top\sigma(p_j)$ y umbrales que se relajan con el tiempo [paper]. Dos muestras se tratan como de la misma clase solo si sus vectores de predicción coinciden con fuerza. Mantiene un clasificador de origen congelado y otro objetivo aprendible ([CVF PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Xia_Adaptive_Adversarial_Network_for_Source-Free_Domain_Adaptation_ICCV_2021_paper.pdf)) [paper].

**U-SFAN (ECCV 2022) y C-SFDA (CVPR 2023). Caen en F1.** U-SFAN pondera solo la entropía de SHOT-IM con $\exp(-H_{Laplace})$, desacoplado en el código publicado. C-SFDA selecciona por fiabilidad sin estructura de clase. Los dos pasarían F3 ([código U-SFAN](https://github.com/roysubhankar/uncertainty-sfda/blob/main/cifar/main_cda.py); [arXiv 2303.17132](https://arxiv.org/abs/2303.17132)) [code / verificado].

**CoWA-JMDS (ICML 2022). Superviviente.**
$$\mathrm{JMDS}_j=\mathrm{LPG}_j\cdot\mathrm{MPPL}_j,\quad \mathrm{LPG}=\tfrac{z_{(1)}-z_{(2)}}{\max(z_{(1)}-z_{(2)})},\quad \mathrm{MPPL}=p_{\hat y}\quad\text{[code]}$$
Ajusta una gaussiana por clase, **con su propia covarianza**, sobre las características objetivo. Una muestra es fiable cuando la brecha entre los dos mayores log-posteriores del GMM es grande **y** la red coincide con la etiqueta del GMM. El peso se calcula bajo `no_grad`. Es el único superviviente que usa covarianzas por clase, pero solo para puntuar, no como objetivo de alineación. Además, el valor por defecto del repositorio es `--coeff LPG`, no `JMDS` ([código CoWA](https://github.com/Jhyun17/CoWA-JMDS/blob/main/image_target_CoWA.py)) [code].

**PLUE (CVPR 2023). Superviviente.** $w=\exp(-H(p)/\log C)$ bajo `no_grad`, con una pseudo-etiqueta refinada por votación blanda de vecinos [code]. Encoge exponencialmente la pérdida de las muestras inciertas, pero nunca por debajo de $e^{-1}\approx0{,}37$ ([código PLUE](https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation/blob/main/train_target.py)) [code].

**DIFO (CVPR 2024), ProDe (ICLR 2025) y DDSR (arXiv 2026).** DIFO adapta *prompts* de CLIP por información mutua y cae en F1. ProDe no se clasificó. DDSR rectifica pseudo-etiquetas con prototipos por clase y fusiona dos maestros por entropía; queda como superviviente en el límite, porque no se verificó si la etapa de prototipos filtra ([arXiv 2311.16510](https://arxiv.org/abs/2311.16510); [arXiv 2603.22908](https://arxiv.org/abs/2603.22908)) [verificado].

## A.4 Maestro–alumno, transformadores y modelos fundacionales

**Self-ensembling (ICLR 2018), DINE (CVPR 2022), Knowledge Adaptation (2017), MT-MTDA (WACV 2021) y DTBS (ECAI 2024). Caen en F1.** Destilan o imponen consistencia hacia un maestro (medio, ensamblado o múltiple) sin alinear estructura por clase. Varios pasarían F3: self-ensembling enmascara si la probabilidad máxima del maestro está por debajo de 0,968 ([arXiv 1706.05208](https://arxiv.org/abs/1706.05208); [arXiv 2104.01539](https://arxiv.org/abs/2104.01539)) [verificado].

**CDTrans (ICLR 2022). Cae en F3 (en el límite).** Asigna etiquetas por centroides, $y_t=\arg\min_kd(c_k,f_t)$, y conserva un par origen–objetivo solo si la pseudo-etiqueta **coincide** con la etiqueta de origen [paper]. Es un filtro de acuerdo, no una puntuación de confianza ([arXiv 2109.06165](https://arxiv.org/abs/2109.06165)) [paper].

**TVT, SSRT, PMTrans, MIC, DAPL y PADCLIP (2022–2023). Caen en F1.** Usan atención consciente de transferibilidad, consistencia ante perturbaciones, mezcla de parches, consistencia enmascarada hacia un maestro EMA (MIC) o pseudo-etiquetas de CLIP con umbral (DAPL $\tau=0{,}6$; PADCLIP $\tau=0{,}4$). MIC, DAPL y PADCLIP pasarían F3 con facilidad, pero no alinean por clase ([arXiv MIC](https://arxiv.org/abs/2212.01322); [CVF PADCLIP](https://openaccess.thecvf.com/content/ICCV2023/html/Lai_PADCLIP_Pseudo-labeling_with_Adaptive_Debiasing_in_CLIP_for_Unsupervised_Domain_ICCV_2023_paper.html)) [verificado].

## A.5 Adaptación en tiempo de prueba

**Tent (ICLR 2021). Cae en F1.** $\min_{\gamma,\beta}-\sum p\log p$ sobre los parámetros afines de batch normalization [memory]. Es minimización de entropía pura, sin estructura de clase ([arXiv 2006.10726](https://arxiv.org/abs/2006.10726)) [verificado].

**EATA, SAR, CoTTA, RoTTA y DeYO (2022–2024). Caen en F1.** EATA pondera con `1/exp(H - E0)` desacoplado [code]. SAR filtra por entropía fiable y usa SAM. CoTTA promedia pesos y aumentos. RoTTA mantiene una memoria equilibrada por clase. DeYO selecciona por entropía y PLPD. **La confianza es común en TTA, pero la alineación condicional por clase no**, y por eso toda la familia muere en F1 ([código EATA](https://github.com/mr-eggplant/EATA/blob/main/eata.py); [arXiv SAR](https://arxiv.org/abs/2302.12400)) [verificado].

**T3A (NeurIPS 2021). Superviviente.** Para cada clase conserva los `filter_K` soportes de menor entropía, $S_k\leftarrow S_k\cup\{f(x)/\|f(x)\|\}$, y sustituye la última capa por sus medias [code]. Reemplaza las plantillas del clasificador por la media de las características de prueba más seguras de cada clase, sin ningún gradiente ([código T3A](https://github.com/matsuolab/T3A/blob/master/domainbed/adapt_algorithms.py)) [code]. TAST, construido sobre T3A, queda en el límite porque no se verificó si conserva el filtro ([arXiv 2207.10792](https://arxiv.org/abs/2207.10792)).

---

# PARTE B. Embudo en notación de CREDA

Esta parte está escrita para el tutor y preparada para convertirse en un diagrama de Archify.

## B.1 Estructura del embudo: 122 → 116 → 66 → 65 → 23

Los recuentos salen de fusionar las cinco notas de investigación, eliminando duplicados (CAN, DSAN, CS-DA, CREDA, BSP, MCC y SHOT aparecen en varias notas). SA y GFK se añadieron después de las notas (2026-10-12), leídos en su fuente primaria.

| Etapa | Entran | Salen | Pasan | Criterio |
|---|---|---|---|---|
| Etapa 0, cribado | 122 | 6 sin clasificar | 116 | Método fuerte o influyente, con mecanismo legible |
| F1, condicional por clase | 116 | 50 | 66 | Alinea $P(Z\mid Y)$ o condiciona la alineación a la clase |
| F2, sin etiquetas objetivo | 66 | 1 | 65 | Usa pseudo-etiquetas, clústeres o predicciones blandas |
| F3, confianza | 65 | 40 + 2 indeterminados | **23** | Pondera o filtra pseudo-etiquetas por confianza o incertidumbre |

**Etapa 0, sin clasificar (6).** Transporte prototípico condicional (Sci. Rep. 2025, página inaccesible); Co-learn y ProDe (examinados solo en parte); dual-teacher retiniano (solo metadatos); candidatos IJCAI 2025 y PASLE de ICLR 2025 (solo fragmentos).

**Salen en F1 (50).** Una línea por grupo, con el motivo común.

| Grupo | Métodos | Motivo de salida |
|---|---|---|
| Adversarios marginales y de píxel (5) | DANN, ADDA, CoGAN, PixelDA, CyCADA | El discriminador no ve la clase; la pérdida semántica de CyCADA preserva la etiqueta pero no alinea por clase |
| Envoltorios adversarios (6) | BSP, GVB, SDAT, ELS, GH/GH++, AFN | No son condicionales por sí mismos (AFN ni siquiera es adversario); sobre CDAN+E heredan su estado |
| Confianza en la variable equivocada (2) | RADA, TADA | Usan la entropía del discriminador de dominio, no del posterior de clase |
| Kernel, momentos y subespacios marginales (10) | TCA, MIDA (Yan), DAN, RTN, CORAL, Deep CORAL, CMD, HoMM, SA, GFK | Solo marginales; CORAL y HoMM son de segundo orden o más, pero sin clase; SA y GFK alinean un subespacio PCA por dominio |
| Información marginal o conjunta (1) | RJSD | Su uso en DA es una conjunta de capas al estilo JAN, sin enrutamiento por clase |
| Pseudo-etiquetado sin alineación (7) | CBST, CRST, MCC, BNM, AdaMatch, CST, U-SFAN (código publicado) | Selección equilibrada o regularización de predicciones; varios pasarían F3 |
| SFDA y maestro–alumno sin clase (7) | C-SFDA, DIFO, self-ensembling, DINE, KD-DA, MT-MTDA, DTBS | Consistencia o destilación sin estructura por clase |
| Transformadores y CLIP (6) | TVT, SSRT, PMTrans, MIC, DAPL, PADCLIP | Atención, mezcla o consistencia; MIC, DAPL y PADCLIP pasarían F3 |
| TTA (6) | Tent, EATA, SAR, CoTTA, RoTTA, DeYO | Entropía o selección sin alineación por clase; EATA, SAR y DeYO pasarían F3 |

El total de 50 resulta de 13 adversarios (5 + 6 + 2), 10 de kernel, momentos y subespacios, 1 de información y 26 de autoentrenamiento, SFDA, transformadores y TTA (7 + 7 + 6 + 6).

**Variantes que vuelven a entrar.** DANN→CDAN y MADA. DAN→DSAN/LMMD. TCA→JDA. HoMM→COT. CDAN→CDAN+E. SHOT→SHOT++. ATDOC-NC→ATDOC-NA. No se encontró ninguna variante condicional publicada de ADDA, Tent, EATA, SAR, CoTTA, MIC, BNM ni MCC.

**Salen en F2 (1).** PRISM (DA semisupervisada, probablemente con objetivo etiquetado).

**Salen en F3 (40) e indeterminados (2).**

| Grupo | Métodos | Motivo de salida |
|---|---|---|
| Adversarios condicionales (15) | MADA, CDAN, MCD, SWD, DADA, CAT†, DAAN, SymNets, f-DAL, MDD, DALN, ToAlign, CGDM†, NOUN/PRONOUN, InAIt† | Enrutamiento blando o crítico en el espacio de predicciones, sin peso de fiabilidad († tentativo: PDF no leído) |
| Familia JDA (8) | JDA, BDA, VDA, DICD, JGSA, ARTL, MEDA, DJP-MMD | Pseudo-etiquetas duras, todas con el mismo peso |
| Kernel conjunto y repesado (2) | JAN, WMMD | Predicción blanda como factor de kernel; pesos de proporción de clase |
| Transporte óptimo (5) | COT, JDOT, DeepJDOT, ETD, MARS | Etiqueta o atención en el coste, sin confianza |
| Divergencia (1) | CS-DA | Predicción blanda en la Gram $L$, sin pesos |
| Autoentrenamiento y SFDA (9) | MSTN, ATDOC-NC, PCL, SHOT, NRC, AaD, G-SFDA, SF(DA)², CDTrans | Centroides o vecindarios sin peso; SHOT calcula el peso y no lo usa; CDTrans filtra por acuerdo |
| **Indeterminados (2)** | Maximización de independencia condicional (arXiv 2203.03212), SIDA | F3 no determinable desde el resumen |

**Supervivientes (23).**

| Subgrupo | Métodos | Mecanismo de confianza en una línea |
|---|---|---|
| **Explícitos, verificados (16)** | CDAN+E | $1+e^{-H}$, suelo 1, GRL en el código oficial |
| | SRADA | Peso creciente en $H$ (invertido) |
| | CAN/CDD | Filtro de distancia $D_0$ y de clase $N_0$ |
| | DCAN/CMMD | Umbral $\max\hat y>0{,}95$ |
| | TPN | Umbral de puntuación $>0{,}6$ |
| | RWOT | Fiabilidad continua $Q$ en el coste de transporte |
| | CREDA | $w=1-\hat H_2/\log C$ en el bloque objetivo |
| | PFAN | Umbral de coseno con calendario $\tau(m)$ |
| | ProDA | Repesado por prototipos y umbral |
| | ATDOC-NA | Confianza del vecindario, desacoplada |
| | SHOT++ | Las $t_k$ de menor entropía por clase |
| | DaC | Máscara $\max p\ge0{,}97$ |
| | CoWA-JMDS | LPG·MPPL del GMM por clase, desacoplado |
| | A2Net | Signo de par con umbrales $\mu(t)$, $l(t)$ |
| | PLUE | $\exp(-H/\log C)$, desacoplado |
| | T3A | Los `filter_K` soportes de menor entropía |
| **Evidencia débil (2)** | BACG (retirado), DARES (software) | Resumen o README |
| **Blandos (3)** | DSAN/LMMD, CPDA, PCT | La probabilidad blanda es a la vez ruta y peso |
| **En el límite (2)** | DDSR, TAST | Mecanismo no verificado del todo |

La lectura del embudo es nítida. **F1 elimina más de dos quintas partes del campo (50 de 116), y con ello toda la TTA salvo T3A y TAST, y todos los transformadores salvo CDTrans.** Al menos once métodos descartados en F1 pasarían F3 sin dificultad (EATA, SAR, DeYO, MIC, DAPL, PADCLIP, AdaMatch, self-ensembling, KD-DA, C-SFDA, U-SFAN): **la confianza es común; la condicionalidad por clase con confianza es rara**.

## B.2 Glosario operativo y símbolos auxiliares

Se usan los símbolos de instancia de la propuesta r29: $\mathbf h=F_\theta(\mathbf x)$ es el embedding de una muestra, $\Phi_\sigma(\mathbf h)\in\mathcal H_I$ su imagen bajo el kernel de instancia $\kappa^I_\sigma$, y $K^s$, $K^t$, $K^{st}$ y $K_c$ son las Gram fuente, objetivo, cruzada y de la clase $c$. Para trasladar métodos que necesitan objetos que esa notación no define, se añaden **símbolos auxiliares**, marcados aquí para que no se confundan con objetos de la propuesta:

$\Phi^s_i\equiv\Phi_\sigma(\mathbf h_i^s)$ se calcula con el codificador de origen **congelado** de la fase 1, y $\Phi^t_j\equiv\Phi_\sigma(\mathbf h_j^t)$ con el codificador objetivo.

$\bar\mu_c^s\equiv\frac1{n_c^s}\sum_{i\in S_c}\Phi^s_i$ y $\bar\mu_c^t\equiv\frac1{n_c^t}\sum_{j\in T_c}\Phi^t_j$ son medias de clase en $\mathcal H_I$. Son símbolos auxiliares.

Los métodos con discriminador o con mapas multilineales, que no admiten $\Phi_\sigma$ de dimensión infinita, operan sobre el embedding finito $\mathbf h$.

$\hat H_1(g)=-\sum_cg_c\ln g_c$ es la entropía de Shannon, para distinguirla del $\hat H_2$ de CREDA.

**Dos identidades [derivado] que se usan en toda la comparación.** La primera reescribe la geometría. Sean $\Sigma_c^s=\frac{1}{\operatorname{tr}K_c^s}\sum_{i\in S_c}\Phi_i^s\otimes\Phi_i^s$ y $\Sigma_c^t=\frac{1}{\operatorname{tr}\tilde K_c^t}\sum_{j\in T_c}(w_j^t)^2\,\Phi_j^t\otimes\Phi_j^t$, operadores de covarianza no centrados de traza unidad sobre $\mathcal H_I$. Entonces
$$\rho_c^s=\|\breve K_c^s\|_F^2=\|\Sigma_c^s\|_{HS}^2,\qquad \rho_c^t=\|\Sigma_c^t\|_{HS}^2,\qquad \rho_c^\times\equiv\frac{\|K_c^{st}D_c^t\|_F^2}{\operatorname{tr}K_c^s\,\operatorname{tr}\tilde K_c^t}=\langle\Sigma_c^s,\Sigma_c^t\rangle_{HS},$$
$$\rho_c^{mix}=\tfrac14\big(\rho_c^s+\rho_c^t+2\rho_c^\times\big)\ \Rightarrow\ J_c=\ln\frac{\rho_c^{mix}}{\sqrt{\rho_c^s\rho_c^t}}=\ln\!\Big(\tfrac12r_c+\tfrac12e^{-D_{CS,c}}\Big),$$
$$r_c=\cosh\!\Big(\frac{H_2(\breve K_c^t)-H_2(\breve K_c^s)}{2}\Big)\ge1,\qquad D_{CS,c}=-\ln\frac{\langle\Sigma_c^s,\Sigma_c^t\rangle_{HS}}{\|\Sigma_c^s\|_{HS}\|\Sigma_c^t\|_{HS}}\ge0 .$$
$D_{CS,c}$ es menos el logaritmo del **coseno de Hilbert–Schmidt entre los operadores de covarianza de clase** de los embeddings de instancia. Equivale a una divergencia de Cauchy–Schwarz entre embeddings de media bajo el kernel $(\kappa^I_\sigma)^2$. La muestra objetivo $j$ entra en $\Sigma_c^t$ con peso proporcional a $(w_j^t)^2$, no a $w_j^t$: el **cuadrado** del peso de confianza.

La segunda reescribe la normalización. Con $U_c=\ln\frac{1+r_c}{2}$ y $L_c=-\ln2$,
$$U_c-J_c=\ln\frac{1+r_c}{r_c+e^{-D_{CS,c}}},\qquad \ell_{\mathrm{glob},c}=\frac{\ln\frac{1+r_c}{r_c+e^{-D_{CS,c}}}}{\ln(1+r_c)}\in\Big[0,\ \frac{\ln(1+1/r_c)}{\ln(1+r_c)}\Big]\subseteq[0,1],$$
y $\ell_{\mathrm{glob},c}=0\iff D_{CS,c}=0$. Ambas identidades se comprobaron numéricamente durante la investigación y no proceden de ninguna publicación. Una precisión: a nivel de instancia $\kappa^I_\sigma$ es gaussiano en $\mathbf h$, así que $(\kappa^I_\sigma)^2$ es gaussiano de ancho $\sigma/\sqrt2$ y la lectura como operadores de covarianza coincide con la de Cauchy–Schwarz de Parzen con ese ancho [derivado].

## B.3 Cada superviviente y cada pariente reescrito en notación de CREDA

Todo lo que sigue es **[derivado]**: cada fórmula es la pérdida del método trasladada a muestras de origen etiquetadas y muestras objetivo sin etiquetar, con los objetos de CREDA cuando existen y con símbolos auxiliares cuando no. Cada entrada termina con la lista de objetos de CREDA que el método **no tiene**.

**Referencia: CREDA con dos codificadores (r29).**
$$\mathcal L=\mathcal L_{src}+\lambda_{glob}\sum_{c\in\mathcal C_{act}}\frac{q_c}{\sum_{c'}n^t_{c'}}\,\ell_{\mathrm{glob},c} .$$

### B.3.1 Parientes parciales pedidos expresamente

**DANN en nuestra notación.** Hay un solo codificador para ambos dominios y un discriminador $D$ sobre $\mathbf h$:
$$\min_{\text{cod},\,\text{cls}}\max_D\ \mathcal L_{src}-\lambda\Big[\tfrac1{N_s}\textstyle\sum_{i\in S}\log D(\mathbf h_i^s)+\tfrac1{N_t}\sum_{j\in T}\log\big(1-D(\mathbf h_j^t)\big)\Big].$$
*Sin contraparte:* $\tilde y_j^t$, $T_c$, $w_j^t$, $D_c^t$, toda Gram, $J_c$, $\ell_{\mathrm{glob},c}$, $q_c$ y el codificador de origen congelado. Tiene un objeto que nosotros no tenemos: el crítico aprendido $D$.

**ADDA en nuestra notación.** La fase 1 es idéntica a la nuestra ($\mathcal L_{src}$ sobre $\mathbf h^s$). En la fase 2 se congelan el codificador de origen y el clasificador, y se entrena solo el codificador objetivo:
$$\min_D-\tfrac1{N_s}\textstyle\sum_i\log D(\mathbf h_i^s)-\tfrac1{N_t}\sum_j\log\big(1-D(\mathbf h_j^t)\big),\qquad \min_{\text{cod}_t}-\tfrac1{N_t}\sum_j\log D(\mathbf h_j^t).$$
*Comparte con CREDA* la topología completa: dos codificadores, origen congelado, objetivo inicializado desde el origen y alineación que solo mueve el lado objetivo. *Sin contraparte:* $\tilde y$, $T_c$, $w$, Gram, $J_c$ y $q_c$. El riesgo que esto deja abierto es el desajuste de clases: el codificador objetivo puede llevar la clase $c$ objetivo a la región de la clase $c'\ne c$ de origen sin que nada lo penalice. El condicionamiento por $T_c$ de CREDA existe precisamente para cerrar ese hueco.

**CDAN+E en nuestra notación.** Hay un solo codificador y el discriminador ve $\mathbf h\otimes g$:
$$\max_D\min_{\text{cod}}\ \mathcal L_{src}-\lambda\Big[\textstyle\sum_i\tilde v^s_i\log D(\mathbf h^s_i\otimes e_{c_i^s})+\sum_j\tilde v^t_j\log\big(1-D(\mathbf h^t_j\otimes g_j^t)\big)\Big],$$
$$v_j^t=1+e^{-\hat H_1(g_j^t)}\in[1+\tfrac1C,2],\qquad \tilde v^t_j=v^t_j/\textstyle\sum_{j'}v^t_{j'}.$$
*Correspondencias:* $v_j^t$ hace el papel de $w_j^t$, pero con entropía de Shannon, suelo $1+1/C$ y, en el código oficial, un gradiente adversario a través de la entropía. Usa la predicción completa $g$ en lugar de $\tilde y$ (condicionamiento blando). *Sin contraparte:* $T_c$ duro, $D_c^t$, Gram, $J_c$, normalización y $q_c$. El estadístico alineado es implícito, el que el discriminador consiga detectar. El tamaño del mapa multilineal, $\dim\mathbf h\cdot C$, crece con $C$.

**DSAN / LMMD en nuestra notación.**
$$\hat d=\frac1{|\mathcal C_{both}|}\sum_{c\in\mathcal C_{both}}\Big\|\bar\mu_c^s-\sum_{j\in T}a_{jc}\Phi^t_j\Big\|^2_{\mathcal H_I},\qquad a_{jc}=\frac{\mathrm{sg}[g^t_{jc}]}{\sum_{j'}\mathrm{sg}[g^t_{j'c}]}.$$
Aquí $\mathrm{sg}$ es la detención de gradiente del código oficial, y $\mathcal C_{both}$ son las clases presentes en $\{c_i^s\}$ y en $\{\tilde y_j^t\}$. En términos de Gram, $\hat d_c=\tfrac1{(n_c^s)^2}\mathbf 1^\top K_c^s\mathbf 1+a_c^\top K^t a_c-\tfrac2{n_c^s}\mathbf 1^\top K^{s,t}_{c,\cdot}a_c$, donde $K^t$ es la Gram objetivo **completa**, no restringida a $T_c$. *Correspondencias:* LMMD no tiene $T_c$ duro, porque toda muestra contribuye a toda clase. Su peso $a_{jc}$ es lineal en la probabilidad, mientras que en CREDA la muestra entra con peso $(w_j^t)^2$. *Sin contraparte:* $\hat H_2$, $D_c^t$, $J_c$, $r_c$, normalización, $q_c$ (salvo el filtro de clases presentes) y los dos codificadores. El estadístico es de **primer orden** (diferencia de medias en $\mathcal H_I$).

**La versión publicada de CREDA (2025) en nuestra notación.** Un solo codificador compartido:
$$\mathcal L_{CREDA}=\mathcal L_{src}-\lambda\sum_{c:\,n_c^s n_c^t>0}\Big[\tfrac12\big(H_2(\breve K_c^s)+H_2(\breve{\tilde K}{}_c^t)\big)-H_2\big(\breve K_c^{mix,\mathrm{CREDA}}\big)\Big],\qquad K_c^{mix,\mathrm{CREDA}}=\begin{bmatrix}K_c^s&K_c^{st}\\(K_c^{st})^\top&D_c^tK_c^tD_c^t\end{bmatrix}.$$
*Correspondencias:* $w$, $D_c^t$, $\tilde K_c^t=D_c^tK_c^tD_c^t$, el enrutamiento por $T_c$ y la forma de $J_c$ son **idénticos** a los de la formulación r29 ([PDF CREDA, Ecs. 15–22](https://mdpi-res.com/d_attachment/mathematics/mathematics-13-02602/article_deploy/mathematics-13-02602.pdf)) [paper]. *Diferencias:* el bloque cruzado es $K_c^{st}$ **sin** $D_c^t$, y la matriz mixta se normaliza por su traza total, sin $Q_c$. Las masas por dominio quedan entonces en $n_c^s:\sum_{j\in T_c}(w_j^t)^2$ con kernel de diagonal unitaria, y la identidad en $D_{CS}$ no se cumple [derivado]. *Sin contraparte:* $Q_c$, $\ell_{\mathrm{glob},c}$, $L_c$, $U_c$, $q_c$ (solo se anulan las clases ausentes del lote) y el codificador congelado.

**Observación nueva [derivado]: la mezcla de la versión publicada de CREDA puede no ser semidefinida positiva.** Con una instancia por dominio, $\kappa(x_s,x_t)=0{,}5$ y $w=0{,}3$, se obtiene $K^{mix}=\begin{bmatrix}1&0{,}5\\0{,}5&0{,}09\end{bmatrix}$, cuyo determinante es $0{,}09-0{,}25<0$. Una matriz así no es la Gram de ningún conjunto de puntos ponderados, y su «entropía» pierde la lectura como operador de densidad. La mezcla de la formulación r29, $\tilde K_c^{mix}=\mathrm{diag}(I,D_c^t)\,K_c\,\mathrm{diag}(I,D_c^t)$ seguida de $Q_c\cdot Q_c$, es una **congruencia** de una Gram y conserva la semidefinición positiva. Esta es una justificación matemática concreta del bloque cruzado $K_c^{st}D_c^t$, que el artículo de CREDA no ofrece.

**CS-DA (Yin et al.) en nuestra notación.** Es el pariente teórico-informacional más cercano, aunque cae en F3:
$$\hat D_{CS}=\ln\tfrac{\mathbf 1^\top K^s\mathbf 1}{N_s^2}+\ln\tfrac{\mathbf 1^\top K^t\mathbf 1}{N_t^2}-2\ln\tfrac{\mathbf 1^\top K^{st}\mathbf 1}{N_sN_t}=-2\ln\cos_{\mathcal H_I}\!\Big(\tfrac1{N_s}\textstyle\sum_i\Phi_i^s,\ \tfrac1{N_t}\sum_j\Phi_j^t\Big),$$
más $D_{CCS}$, construida con una Gram de etiquetas $L_{jj'}=\kappa(g_j^t,g_{j'}^t)$. *Correspondencias:* comparte la familia Cauchy–Schwarz. Pero CS-DA mide el coseno entre **embeddings de media** (primer orden), es marginal en su primer término y condiciona a $p(\hat y\mid\Phi)$ en el segundo. CREDA mide el coseno entre **operadores de covarianza por clase** (segundo orden) y condiciona a $p(\Phi\mid y)$. *Sin contraparte:* $T_c$, $w$, $D_c^t$, $r_c$, normalización y $q_c$. Usa dos clasificadores al estilo MCD.

### B.3.2 Supervivientes explícitos verificados

**SRADA.** $\mathcal L=\mathcal L_{src}-\lambda\sum_xu(\hat H_1(g))\ell_d(D(\mathbf h))+\mathcal L_{tri}$, con $u$ **creciente** y $\mathcal L_{tri}=\sum\max\big(0,\|\Phi^t_a-\Phi_p\|^2-\|\Phi^t_a-\Phi_n\|^2+m\big)$, donde los positivos comparten $\tilde y$. La forma exacta de $u$ es [UNVERIFIED]. *Lo opuesto a nosotros:* $u$ crece con la incertidumbre, mientras que $w_j^t$ decrece. *Sin contraparte:* Gram, $J_c$ y $q_c$. *Tiene algo que nos falta:* la tripleta, que separa clases.

**CAN / CDD.** Las etiquetas $\tilde y_j^{t}$ salen de un k-means esférico sobre $\{\Phi^t_j\}$, inicializado en $\bar\mu_c^s$ (no de $\mathfrak a(g_j^t)$). El filtro es $\tilde T_c=\{j\in T_c:\ \tfrac12(1-\cos(\Phi^t_j,O^t_c))<D_0\}$, equivalente a un peso binario $w^{CAN}_j=\mathbb 1[\cdot]$, y la compuerta de clase es $\mathbb 1[|\tilde T_c|>N_0]$:
$$\hat D^{cc'}=\tfrac1{(n_c^s)^2}\mathbf 1^\top K_c^s\mathbf 1+\tfrac1{|\tilde T_{c'}|^2}\mathbf 1^\top K_{c'}^t\mathbf 1-\tfrac2{n_c^s|\tilde T_{c'}|}\mathbf 1^\top K^{st}_{cc'}\mathbf 1=\|\bar\mu_c^s-\bar\mu_{c'}^t\|^2_{\mathcal H_I},$$
$$\mathcal L_{CDD}=\tfrac1M\textstyle\sum_c\hat D^{cc}-\tfrac1{M(M-1)}\sum_{c\ne c'}\hat D^{cc'} .$$
*Correspondencias:* $\mathbb 1[|\tilde T_c|>N_0]$ cumple el papel de $q_c$, pero como cuenta dura y no como masa de confianza. El primer término es la versión de primer orden de nuestro $J_c$. *Tiene algo que nos falta:* $-\hat D^{cc'}$, la **repulsión interclase**. *Sin contraparte:* $w$ continuo, $\hat H_2$, $r_c$ y normalización por rango. Usa un codificador con BN por dominio, multikernel, y etiquetas sin gradiente por construcción.

**DCAN / CMMD.** Con $L^s=Y^sY^{s\top}$ (one-hot de $c_i^s$) y $L^t$ construida con $\tilde y_j^t$ solo para $j$ tal que $\max_cg^t_{jc}>0{,}95$, y $K$ formado con $\kappa^I_\sigma$:
$$\mathcal L_{CMMD}=\mathrm{Tr}(K^s\tilde L_s^{-1}L^s\tilde L_s^{-1})+\mathrm{Tr}(K^t\tilde L_t^{-1}L^t\tilde L_t^{-1})-2\,\mathrm{Tr}(L^{ts}\tilde L_s^{-1}K^{st}\tilde L_t^{-1}).$$
*Correspondencias:* el umbral $0{,}95$ es una compuerta binaria por muestra, frente a nuestro $w$ continuo. El condicionamiento lo hace el kernel de etiquetas en lugar de un enrutamiento explícito por $T_c$. *Sin contraparte:* $\hat H_2$, $J_c$, normalización y $q_c$. *Tiene algo que nosotros evitamos:* la inversa $(L+\lambda I)^{-1}$ y el regularizador $\lambda$.

**TPN.** $\tilde y_j^t=\arg\min_c\|\Phi^t_j-\bar\mu^s_c\|$, aceptada solo si la puntuación supera 0,6, y $\bar\mu_c^{st}$ es la media sobre $S_c\cup T_c$:
$$L_G=\tfrac1C\textstyle\sum_c\big[\|\bar\mu_c^s-\bar\mu_c^t\|^2+\|\bar\mu_c^s-\bar\mu_c^{st}\|^2+\|\bar\mu_c^t-\bar\mu_c^{st}\|^2\big]_{\mathcal H_I}+\text{KL simétrica por muestra}.$$
*Correspondencias:* el prototipo mixto $\bar\mu_c^{st}$ es el análogo de primer orden del bloque mixto $\breve K^{mix}_c$. *Sin contraparte:* $w$ continuo, $r_c$, normalización y $q_c$.

**RWOT.** Hay un plan $\gamma$ entre muestras de origen y muestras objetivo:
$$Z_{ij}=\|\Phi_i^s-\Phi_j^t\|^2_{\mathcal H_I}\big(1-Q(j,c_i^s)\big),\quad Q(j,c)\propto d_{\mathcal A}(c)\,\mathrm{softmax}_c\big(-d(\Phi^t_j,\bar\mu^s_c)\big)+(2-d_{\mathcal A}(c))\,g^{t,\tau}_{jc}.$$
*Correspondencias:* $Q(j,\cdot)$ es una fiabilidad **por clase** y continua, más rica que nuestro $w_j^t$ escalar. Usa $g$ completa. *Sin contraparte:* $T_c$ duro, Gram por clase, $J_c$ y $q_c$. El coste $O(N_sN_t)$ del plan introduce sesgo de transporte en minilotes pequeños.

**PFAN.** $\tilde y_j^t=\arg\max_c\cos(\Phi^t_j,\bar\mu^s_c)$, con $w^{PFAN}_j=\mathbb 1[\cos\ge\tau(m)]$ y $\tau(m)$ creciente con el calendario; además $\mathcal L_{apa}=\sum_cd(\bar\mu_c^s,\bar\mu_c^{t,\mathrm{EMA}})$. *Tiene algo que nos falta:* un **calendario de umbral**. *Sin contraparte:* $\hat H_2$, Gram y $J_c$. Alinea primer orden.

**ProDA.** $\tilde y_j^t=\arg\max_c\big[\omega_{jc}\,g^{t,0}_{jc}\big]$, con $\omega_{jc}=\mathrm{softmax}_c\big(-(\|\Phi^t_j-\eta_c\|-\min_{c'}\|\Phi^t_j-\eta_{c'}\|)T\big)$ bajo `no_grad`, y CE sobre las muestras aceptadas. *Correspondencias:* combina $g$ completa con distancias a prototipos para **rectificar la etiqueta**, no para alinear. *Sin contraparte:* cualquier estadístico de alineación.

**ATDOC-NA.** $\mathcal L=\sum_j\mathrm{sg}[\bar w_j]\,\mathrm{CE}(g^t_j,\tilde y_j)/\mathrm{sg}[\sum_j\bar w_j]$, con $\bar w_j=\max_c\frac1K\sum_{k\in\mathcal N(j)}m_{kc}$. *Correspondencias:* es una media ponderada $\sum w\ell/\sum w$ con peso **y denominador desacoplados**. *Sin contraparte:* Gram, $J_c$ y el lado de origen.

**SHOT++.** Con clasificador de origen congelado y codificador objetivo propio: $\mathcal L=\mathbb E_j[\hat H_1(g^t_j)]-\hat H_1(\bar g^t)+\beta\,\mathrm{CE}(g^t_j,\tilde y^{cent}_j)$, y en la segunda etapa $E_c=\{t_c\ \text{muestras de }T_c\ \text{con menor }\hat H_1\}$ alimenta MixMatch. *Correspondencias:* $E_c$ es una compuerta por rango dentro de cada clase. El congelamiento afecta a la hipótesis de origen, no a un segundo codificador. *Tiene algo que nos falta:* el término de diversidad $-\hat H_1(\bar g^t)$, que es anticolapso. *Sin contraparte:* Gram, $J_c$ y el lado de origen (es sin fuente).

**DaC.** $\mathcal M=\{j:\max_cg^t_{jc}\ge0{,}97\}$ (máscara desacoplada), prototipos sobre $\mathcal M$ y contraste más EMMD/LMMD entre $\mathcal M$ y su complemento. *Correspondencias:* compuerta binaria por muestra y alineación de primer orden entre dos subconjuntos objetivo. *Sin contraparte:* $w$ continuo, $J_c$, $q_c$ y el lado de origen.

**CoWA-JMDS.** Un GMM por clase sobre $\{\Phi^t_j\}$ con medias $m_c$ y covarianzas $\Sigma^{GMM}_c$ (en una proyección finita), $\tilde y^{GMM}_j=\arg\max_c\gamma_{jc}$ y $w^{JMDS}_j=\mathrm{sg}\big[\mathrm{LPG}_j\cdot g^t_{j,\tilde y_j}\big]$ multiplicando una CE o KL con *mixup*. *Correspondencias:* es el único superviviente con una **covarianza por clase**, pero la usa para **puntuar**, no como objetivo de alineación entre dominios. Nuestro $\Sigma_c^t$ se compara con $\Sigma_c^s$. *Sin contraparte:* el lado de origen (sin fuente), $J_c$ y $q_c$.

**A2Net.** $\gamma_{jj'}=\mathbb 1[\langle g_j,g_{j'}\rangle>\mu(t)]-\mathbb 1[\langle g_j,g_{j'}\rangle<l(t)]$ sobre pares de muestras objetivo, con un clasificador de origen congelado y otro objetivo aprendible. *Tiene algo que nos falta:* **umbrales que se relajan** con el tiempo y pares negativos. *Sin contraparte:* Gram por clase, $J_c$ y el lado de origen.

**PLUE.** $\mathcal L=\frac1{N_t}\sum_j\mathrm{sg}\big[e^{-\hat H_1(\bar g_j)/\ln C}\big]\,\mathrm{CE}(g^t_j,\tilde y^{kNN}_j)$. *Correspondencias:* es un peso entrópico con suelo $e^{-1}$, frente a nuestro $w\in[0,1]$ con cero en la uniforme. *Sin contraparte:* alineación de distribuciones.

**T3A.** $\bar\mu^{T3A}_c=\mathrm{mean}\{\Phi^t_j/\|\Phi^t_j\|:\ j\in T_c,\ \hat H_1(g_j)\ \text{entre los filter\_K menores}\}$, y se predice con $\cos(\Phi^t,\bar\mu^{T3A}_c)$. No hay gradiente. *Sin contraparte:* pérdida alguna; es un reajuste del clasificador.

### B.3.3 Supervivientes blandos, de evidencia débil y en el límite

**CPDA.** $\sum_c\pi_t(c)\|\bar\mu_c^s-\sum_ja_{jc}\Phi^t_j\|^2$, con $a_{jc}$ desacoplado y $\tau=0$; es LMMD con pesos de prior de clase $\pi_t(c)$. *Tiene algo que nos falta:* una cota de riesgo explícita. **PCT.** Transporte blando entre $\Phi^t_j$ y prototipos fijos (los pesos del clasificador), con priors $p(\mu_c)$ estimados por EM. *Tiene algo que nos falta:* corrección del desplazamiento de etiquetas. **DARES.** Opera sobre píxeles; una alineación Rényi-2 condicional «endurecida»; peso $w$ por píxel; $\beta\mathcal L_{ac}$ con suelos $H_2(\breve K_c^t)\ge\eta$; $\gamma\mathcal L_{rep}$ de repulsión interclase con bisagra de margen; región de confianza GradNorm y un operador de muestreo estratificado $\Phi_c$ ([DARES](https://github.com/liturriago/DARES)) [verificado: README; formas exactas UNVERIFIED]. *Tiene dos cosas que nos faltan:* el **suelo anticolapso** y la **repulsión**. **BACG**: retirado. **DDSR y TAST**: véase la Parte A.

## B.4 Tabla comparativa final: supervivientes y parientes frente a CREDA

«Orden» indica el orden del estadístico en el espacio de embeddings. «Ruta» indica cómo se asignan las muestras objetivo a las clases. «Det.» indica si el peso está desacoplado del grafo de gradiente.

| Método | Orden del estadístico | Ruta | Peso de confianza (fórmula; det.) | Nivel | Codificadores | Normalización | Separación interclase |
|---|---|---|---|---|---|---|---|
| **CREDA (r29)** | **2.º: coseno HS entre $\Sigma_c^s$ y $\Sigma_c^t$** | Dura ($\mathfrak a$) | $1-\hat H_2/\ln C\in[0,1]$, entra como $w^2$; **no** det. | Instancia | **Dos, origen congelado** | $\ell_{\mathrm{glob},c}\in[0,1]$ por rango alcanzable; $\omega_c\propto q_c$ | **No** |
| CREDA (2025) | 2.º sin forma cerrada (mezcla no equilibrada) | Dura | $1-\hat H_2/\log C$ solo en el bloque $tt$; no declarado | Instancia | Uno | Ninguna (suma 1/1) | No |
| DARES | Rényi-2 «endurecida» | Dura | Rényi-2 por píxel; no declarado | Píxel | Uno | No declarada | **Sí** (bisagra) |
| CDAN+E | Implícito (crítico) | Blanda ($g$ completa) | $1+e^{-\hat H_1}$, suelo 1; GRL (oficial) / det. (TLlib) | Instancia | Uno | Por dominio | No |
| SRADA | Implícito | Blanda | Creciente en $H$ (invertido); UNVERIFIED | Instancia | Uno | — | Tripleta |
| CAN/CDD | 1.º (medias en RKHS) | Dura (k-means) | Binario $D_0$, clase $N_0$; sin gradiente | Instancia | Uno + BN por dominio | $1/M$ | **Sí** |
| DCAN/CMMD | Operador condicional $Z\mid Y$ | Por kernel de etiquetas | Binario $\max g>0{,}95$ | Instancia | Uno | Regularizador $\lambda$ | No |
| TPN | 1.º (prototipos) | Dura (prototipo más cercano) | Binario $>0{,}6$ | Instancia | Uno | $1/C$ | No |
| RWOT | Transporte (coste) | Blanda | $Q(j,c)$ continuo; no declarado | Instancia | Uno | Plan | No |
| DSAN/LMMD | 1.º | Blanda | $g_{jc}/\sum g$; **det.** | Instancia | Uno | $1/|\mathcal C_{both}|$ | No |
| CPDA | 1.º | Blanda | Blanda; **det.**; $\tau$ opcional | Instancia (trayectoria) | Uno | $\pi_t(c)$ | No |
| PCT | Transporte a prototipos | Blanda | Blanda; prototipos **fijos** | Instancia | Uno | Priors EM | No |
| PFAN | 1.º | Dura | Binario con calendario $\tau(m)$ | Instancia | Uno | — | No |
| ProDA | Rectificación (sin alineación) | Dura | Prototipo × $g$; **det.** | Píxel | Uno | — | No |
| ATDOC-NA | Ninguno (CE) | Dura | Vecindario; **det.** (peso y denominador) | Instancia | Uno | $\sum w\ell/\sum w$ | No |
| SHOT++ | Ninguno (IM + MixMatch) | Dura | Rango por clase | Instancia | Hipótesis congelada | — | Diversidad |
| DaC | 1.º (EMMD/LMMD interna) | Dura | Binario $0{,}97$; **det.** | Instancia | Uno | — | Contraste |
| CoWA-JMDS | 2.º **solo para puntuar** | Dura (GMM) | LPG·MPPL; **det.** | Instancia | Uno (sin fuente) | — | No |
| A2Net | Pares | Por pares | Umbrales $\mu(t)$, $l(t)$ | Instancia | Clasificador de origen congelado | — | Pares negativos |
| PLUE | Ninguno (CE) | Dura (kNN) | $e^{-H/\ln C}$, suelo $e^{-1}$; **det.** | Instancia | Uno | Media | No |
| T3A | Prototipos sin gradiente | Dura | Filtro de entropía | Instancia | Congelado | — | No |
| CS-DA (F3 ✗) | 1.º (coseno de medias) | Blanda (Gram $L$) | Ninguno | Instancia | Uno | Logaritmo de cociente | No |
| ADDA (F1 ✗) | Implícito | Ninguna | Ninguno | Instancia | **Dos, origen congelado** | — | No |
| DANN (F1 ✗) | Implícito | Ninguna | Ninguno | Instancia | Uno | — | No |

En cada columna, CREDA tiene un vecino: CAN y DCAN por la compuerta, CoWA por el segundo orden y ADDA por los codificadores. **En ninguna otra fila coinciden a la vez alineación de segundo orden y origen congelado.** En la columna de separación interclase, CREDA es de los pocos con un «No» que no compensa por otra vía (CAN, DARES, SRADA, A2Net, DaC y SHOT++ tienen algún mecanismo).

## B.5 Por qué ganamos, y dónde no

### Lo que se hereda de la versión publicada de CREDA y no puede reclamarse

El peso $w_j^t=1-\hat H_2(g_j^t)/\ln C$ es la Ec. 18–19 de CREDA. El enrutamiento duro por $\mathfrak a(g)$ es su Ec. 15–17. La ponderación $\tilde K_c^t=D_c^tK_c^tD_c^t$ es su Ec. 20 ($K^t\odot ww^\top$). La forma $J_c=\tfrac12(H_2^s+H_2^t)-H_2^{mix}$ es su Ec. 21. Todo esto está [paper] en el PDF ([PDF CREDA](https://mdpi-res.com/d_attachment/mathematics/mathematics-13-02602/article_deploy/mathematics-13-02602.pdf)). También se hereda $\alpha=2$. Hay dos frases que el capítulo no debe escribir: «la ponderación por confianza es nueva» (la versión publicada de CREDA, DSAN, DARES y los diez supervivientes de autoentrenamiento la tienen) y «primera DA condicional con Rényi» (lo es la versión publicada de CREDA). Tampoco debe heredar la etiqueta de «información mutua» que CREDA da a $\tilde I_2$: $K^s$ y $K^t$ son Gram de conjuntos disjuntos y no vistas emparejadas, así que $\tilde I_2$ es una brecha de tipo Jensen–Rényi y no una información mutua en el sentido de la Ec. 11 del propio artículo [derivado]. Conviene señalar además que la Ec. 23 de CREDA, tal como está impresa, tiene un signo inconsistente (la CE sin el menos), mientras que el repositorio minimiza $\mathrm{CE}-\lambda\tilde I_2$ ([repositorio](https://github.com/Daprosero/Domain_Adaptation)) [verificado].

La base empírica heredada es débil. CREDA usa una sola semilla (42), presenta «±» como desviación entre lotes de prueba, toma un subconjunto de 10 clases de Office-31 y compara solo con DANN, ADDA y CDAN+E, sin ningún método de la familia MMD condicional. Pierde frente a DANN en ImageCLEF con ResNet-50 (81,67 frente a 83,15) y frente a la línea base en Digits con ViT-Tiny (47,23 frente a 51,28) ([PDF CREDA, Tablas 6–8](https://mdpi-res.com/d_attachment/mathematics/mathematics-13-02602/article_deploy/mathematics-13-02602.pdf)) [paper]. La formulación con dos codificadores no puede apoyarse en esos números como evidencia frente a CAN, DSAN o CS-DA.

### Lo que es genuinamente nuestro, con su consecuencia matemática

**La mezcla equilibrada por congruencia.** Con $Q_c$ y el bloque cruzado $K_c^{st}D_c^t$, $J_c$ es **exactamente** $\ln(\tfrac12r_c+\tfrac12e^{-D_{CS,c}})$, lo que separa la alineación ($D_{CS,c}$) del desajuste de pureza ($r_c$). En la forma publicada de CREDA, ambos quedan confundidos con la razón de masas $n_c^s:\sum w^2$ [derivado]. La congruencia garantiza además que la mezcla es semidefinida positiva, algo que el bloque cruzado sin ponderar de la versión publicada no garantiza (contraejemplo en B.3.1) [derivado].

**La normalización r29.** $\ell_{\mathrm{glob},c}=0\Leftrightarrow D_{CS,c}=0\Leftrightarrow\Sigma_c^s=\Sigma_c^t$. Si $(\kappa^I_\sigma)^2$ es característico, eso equivale a la igualdad de las distribuciones condicionales de los embeddings de la clase $c$, por el mismo argumento del Teorema 6 de RJSD ([arXiv 2305.16446](https://arxiv.org/abs/2305.16446)) [derivado]. Esto corrige el defecto de signo de la brecha de Jensen–Rényi con $\alpha>1$, que puede ser negativa: con $p=(1,0,0)$ y $q=(\tfrac34,\tfrac18,\tfrac18)$ se obtiene $\mathrm{JR}_2=-0{,}0054$ [derivado]. Además coloca todas las clases en una escala común $[0,1]$. Ningún superviviente normaliza cada clase por su propio rango alcanzable.

**La compuerta $q_c$.** $\omega_c=q_c/\sum_{c'}n^t_{c'}$ pondera cada clase por su **masa de confianza**. CAN y SHOT usan cuentas duras ($N_0$ o un umbral de recuento) y la versión publicada de CREDA solo anula las clases ausentes del lote ([arXiv 1901.00976](https://arxiv.org/pdf/1901.00976); [código SHOT](https://raw.githubusercontent.com/tim-learn/SHOT/master/object/image_target.py)) [paper/code].

**Dos codificadores con el de origen congelado.** Con el origen congelado, $\Sigma_c^s$ y $H_2(\breve K_c^s)$ son constantes para una composición de lote dada, y solo se mueve el lado objetivo. Eso elimina el grado de libertad que tiene un codificador compartido de «mover ambos dominios hacia un punto común», que es la puerta del colapso conjunto. Los precedentes son ADDA en adversarios, JGSA y PCT en discrepancia, y SHOT, CoWA y A2Net en SFDA, pero ninguno lo combina con un término condicional ponderado ([CVF ADDA](https://openaccess.thecvf.com/content_cvpr_2017/html/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.html); [arXiv JGSA](https://arxiv.org/pdf/1705.05498)) [verificado].

**Alineación de segundo orden de operadores de covarianza por Cauchy–Schwarz.** Todos los supervivientes alinean primer orden (medias, prototipos, embeddings de media) o usan la covarianza solo para puntuar (CoWA). CS-DA mide el coseno entre embeddings de media. CREDA mide el coseno de Hilbert–Schmidt entre $\Sigma_c^s$ y $\Sigma_c^t$ [derivado]. Hay que decir con honestidad qué compra esto. Con $\alpha=2$, el contenido informativo es el de los tres términos de la $\mathrm{MMD}^2$ bajo $(\kappa^I_\sigma)^2$, es decir, $J_c=\ln\frac{2(\rho^s+\rho^t)-\mathrm{MMD}^2_{(\kappa^I_\sigma)^2}}{4\sqrt{\rho^s\rho^t}}$, y no «más estructura» que una MMD bajo el kernel al cuadrado [derivado]. Lo que sí aporta frente a una MMD de primer orden tiene tres partes. La primera es la **invariancia a la escala**: $\Sigma_c^t$ tiene traza unidad, así que el codificador objetivo no puede reducir la pérdida encogiendo o inflando las normas de $\Phi^t$, mientras que $\|\bar\mu^s_c-\bar\mu^t_c\|^2$ sí depende de esas normas. La segunda es una **geometría de coseno** entre operadores. La tercera es la **supresión cuadrática** de las muestras dudosas: el peso $(w_j^t)^2$ en $\Sigma_c^t$ castiga una muestra mal enrutada con baja confianza más fuerte que el peso lineal de LMMD [derivado]. Que estas propiedades se traduzcan en exactitud es una pregunta empírica.

**Sin juego adversario.** CREDA es minimización pura, sin crítico, sin GRL y sin calendarios de $\lambda$. La inestabilidad del entrenamiento adversario está documentada en varios trabajos: discriminadores sobreconfiados y ruido de etiquetas de dominio ([ICLR ELS](https://iclr.cc/virtual/2023/poster/11317)), mínimos planos de la pérdida adversaria que dañan la generalización ([ICML SDAT](https://icml.cc/virtual/2022/poster/17769)), discriminadores que se degradan al alinearse las características ([CVF RADA](https://openaccess.thecvf.com/content/ICCV2021/html/Jin_Re-Energizing_Domain_Discriminator_With_Sample_Relabeling_for_Adversarial_Domain_Adaptation_ICCV_2021_paper.html)) y conflicto entre gradientes de alineación y de tarea ([arXiv GH](https://arxiv.org/abs/2408.00288v1)) [verificado]. Esta es la ventaja estructural más defendible frente a CDAN+E, el superviviente adversario canónico.

### Lo que otros tienen y a nosotros nos falta

**Repulsión interclase.** CDD resta $\frac1{M(M-1)}\sum_{c\ne c'}\hat D^{cc'}$ y DARES usa una repulsión con bisagra de margen ([arXiv 1901.00976](https://arxiv.org/pdf/1901.00976); [DARES](https://github.com/liturriago/DARES)) [paper/verificado]. $J_c$ es puramente intraclase. Fuera de la CE en el origen y del enrutamiento duro, nada en $\mathcal L_{glob}$ penaliza que $\Sigma_c^t$ y $\Sigma_{c'}^t$ se solapen, ni que $\Sigma_c^t$ se acerque a $\Sigma_{c'}^s$ con $c'\ne c$.

**Enrutamiento blando.** LMMD reparte cada muestra entre todas las clases. El argmax de CREDA descarta la masa no máxima de $g$ ([lmmd.py](https://raw.githubusercontent.com/jindongwang/transferlearning/master/code/DeepDA/loss_funcs/lmmd.py)) [code].

**Pesos desacoplados.** En todos los supervivientes revisados en código el peso está desacoplado: LMMD, CPDA, PCT (prototipos fijos), ATDOC-NA, CoWA, PLUE, DaC, ProDA, T3A y la versión TLlib de CDAN, además de MCC, EATA y U-SFAN fuera del embudo ([código CoWA](https://github.com/Jhyun17/CoWA-JMDS/blob/main/image_target_CoWA.py); [código PLUE](https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation/blob/main/train_target.py)) [code]. **Ninguna fuente revisada mantiene diferenciable un peso de confianza sobre una pérdida de alineación que se minimiza directamente.** CDAN no es precedente: allí el extractor maximiza la pérdida ponderada a través de la GRL, así que el camino del peso premia la certeza; sin adversario, el signo se invierte [derivado].

**Umbrales adaptativos.** PFAN baja su umbral con $\tau(m)$, A2Net relaja $\mu(t)$ y $l(t)$, y SoftMatch y FlexMatch adaptan por clase ([arXiv PFAN](https://arxiv.org/abs/1811.08585)) [paper]. El $w$ de CREDA tiene escala fija. Al inicio de la fase 2, si las predicciones objetivo son casi uniformes, todo $w\approx0$ y los términos con compuerta se apagan [derivado].

**Suelo anticolapso.** DARES impone $H_2\ge\eta$ ([DARES](https://github.com/liturriago/DARES)) [verificado: README]. Es una defensa directa contra el defecto de $r_c$ que se describe en el siguiente apartado.

**Término marginal y corrección del desplazamiento de etiquetas.** CS-DA, la familia JDA y DAAN añaden un término marginal. WMMD, InAIt, MARS y PCT corrigen las proporciones de clase. CREDA no tiene término marginal, así que toda su alineación depende de la calidad del enrutamiento al principio. Su única defensa frente al desplazamiento de etiquetas es $\omega_c\propto q_c$.

### Los defectos abiertos, en forma matemática

**$w$ no está desacoplado.** En el término global, $\partial\omega_c/\partial w_j^t=1/\sum n^t>0$ con $\ell_{\mathrm{glob},c}\ge0$, así que reducir los $w$ de las clases peor alineadas reduce su contribución, y bajar $w$ significa volver $g_j^t$ más uniforme. El mismo atajo aparece en cualquier media ponderada $L=\sum_jw_j\ell_j/(\sum_jw_j+\varepsilon)$ con $w$ diferenciable: $\partial L/\partial w_j=(\ell_j-L)/(\sum w+\varepsilon)$, así que el optimizador baja $w_j$ en toda muestra cuya pérdida supera la media ponderada. **Ese camino premia la incertidumbre** [derivado]. La corrección es desacoplar, como hace todo precedente. Si se quiere diferenciabilidad, hay dos alternativas: un suelo ($1+w$, como CDAN+E y MCC) o un término de diversidad de tipo SHOT.

**$r_c$ conserva el gradiente en el numerador.**
$$\frac{\partial}{\partial r_c}\ln\frac{1+r_c}{r_c+e^{-D_{CS,c}}}=\frac1{1+r_c}-\frac1{r_c+e^{-D_{CS,c}}}\le0,$$
con igualdad solo si $D_{CS,c}=0$. Mientras la clase no está alineada, la pérdida baja si **aumenta el desajuste de pureza**, y el numerador tiende a 0 cuando $r_c\to\infty$, con independencia de $D_{CS,c}$ [derivado; verificado numéricamente en las notas: $\partial_r=-0{,}157$, y el numerador cae de 0,274 a 0,004 al multiplicar $r$ por 100]. Con el origen congelado, $\rho_c^s$ es fijo y la única vía es que el codificador objetivo **colapse o disperse** la clase. Las correcciones son desacoplar $r_c$ también en el numerador, usar directamente $D_{CS,c}$ o imponer un suelo como el de DARES.

**$\alpha=2$ no tiene optimalidad publicada.** Se trata en B.6.4.

### El veredicto, desde las matemáticas

CREDA **no gana por condicionalidad ni por confianza**: hay trece años de MMD condicional y 23 supervivientes con confianza. Gana, en sentido formal, porque es el único método que reúne tres propiedades: (i) alinea operadores de covarianza por clase con una distancia de coseno invariante a la escala; (ii) mueve solo el lado objetivo frente a una referencia de origen fija; y (iii) tiene una pérdida acotada en $[0,1]$ que es nula exactamente en la alineación. **No gana, o puede perder**, en cuatro frentes: separación entre clases (CDD, DARES), robustez del arranque (sin término marginal y con $w\approx0$ al inicio), robustez ante el desplazamiento de etiquetas, y en el gradiente, donde dos atajos ($w$ y $r$) premian soluciones degeneradas. Hasta que esos dos atajos se corrijan, ningún resultado empírico de CREDA podrá atribuirse con limpieza a sus rasgos genuinos.

## B.6 Las cinco peticiones del tutor

### B.6.1 Estado del arte de la MMD condicional no supervisada

Existe y es madura. En la vertiente superficial está **JDA** (ICCV 2013) y su familia: BDA, VDA, DICD, ARTL, MEDA, DJP-MMD y JGSA. En la profunda, **CDD/CAN** (CVPR 2019), **LMMD/DSAN** (TNNLS 2021), **WMMD** (CVPR 2017), **CMMD/DCAN** (Pattern Recognition) y **TPN** (CVPR 2019). En 2026 aparece **CPDA**, que incluye MMD, CORAL y HoMM como casos restringidos y aporta una cota de riesgo ([arXiv 1911.02685](https://arxiv.org/pdf/1911.02685); [arXiv 1901.00976](https://arxiv.org/pdf/1901.00976); [arXiv 2106.09388](https://arxiv.org/pdf/2106.09388); [arXiv 2006.07776](https://arxiv.org/pdf/2006.07776); [arXiv 2608.09193](https://arxiv.org/pdf/2608.09193)) [paper]. Todas son, con distinta ponderación, instancias de $\sum_c\|\bar\mu^s_c-\bar\mu^t_c\|^2$; DCAN lo es a nivel de operador condicional. Todas usan un solo codificador y trabajan sobre instancias. De ellas, CAN, DCAN y TPN superan F3 con filtros binarios, y LMMD y CPDA con pesos blandos.

### B.6.2 Embeddings de Hilbert frente a medidas de información

| Medida | Objeto | Estimador | ¿Divergencia válida? | Coste | Fallos | Fuente |
|---|---|---|---|---|---|---|
| MMD | $\|\mu_P-\mu_Q\|^2_{\mathcal H}$ (1.er orden en $\phi$) | U-estadístico | Sí, con kernel característico | $O(n^2)$ | Ancho de banda; depende de la escala | [JMLR](https://jmlr.org/papers/v13/gretton12a.html) [memory] |
| HSIC / MIDA | Dependencia con la variable de dominio | $\frac1{n^2}\mathrm{tr}(KHLH)$ | Caracteriza independencia | $O(n^2)$ | Mide dependencia, no discrepancia | [arXiv 1603.04535](https://www.arxiv.org/pdf/1603.04535) [verificado] |
| CMMD | Operador $Z\mid Y$ | Inversa de Gram regularizada | Sí, condicional | $O(n^3)$ | Requiere $\lambda$ | [arXiv 2006.07776](https://arxiv.org/pdf/2006.07776) [paper] |
| CORAL | Covarianza finita, marginal | $\|C_S-C_T\|_F^2$ | No (solo 2.º momento) | $O(d^2)$ | Necesita características finitas | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/10306) [verificado] |
| KL / JS (DANN), $f$-DAL | Densidades implícitas | Crítico variacional | Sí | Red crítica | Min–max inestable | [NVIDIA fDAL](https://research.nvidia.com/labs/toronto-ai/fDAL) [verificado] |
| Rényi cuadrática (ITL) | $-\log\int p^2$ | Sumas de Parzen | Es entropía, no divergencia | $O(n^2)$ | Ancho de banda | [notas ITL](https://www.dca.fee.unicamp.br/~lboccato/topico_12_information_theoretic_learning.pdf) [verificado] |
| CS-DA | Coseno entre embeddings de media | Logaritmos de sumas de kernel | Sí | $O(n^2)$ | «Más ajustada que KL» solo para gaussianas | [arXiv 2405.19978](https://arxiv.org/pdf/2405.19978) [paper] |
| Rényi matricial $S_\alpha$ | Espectro de la Gram | $\frac1{1-\alpha}\log\mathrm{tr}(A^\alpha)$ | Es entropía | $O(n^2)$ con $\alpha=2$; $O(n^3)$ en general | Pocas muestras por clase | [arXiv 1211.2459](https://arxiv.org/abs/1211.2459) [verificado] |
| Jensen–Rényi (CREDA) | $S(\text{mezcla})-\text{media}$ | Gram en bloques | **No** con $\alpha=2$ (puede ser negativa) | $O(n^2)$ | Signo; mezcla no PSD | [PDF CREDA](https://mdpi-res.com/d_attachment/mathematics/mathematics-13-02602/article_deploy/mathematics-13-02602.pdf) [paper]; [derivado] |
| RJSD | von Neumann entre operadores | Gram en bloques | Sí: nula si y solo si $P=Q$ con $\kappa^2$ característico | $O(n^3)$ | Coste espectral | [arXiv 2305.16446](https://arxiv.org/abs/2305.16446) [paper] |
| **$\ell_{\mathrm{glob},c}$ (r29)** | **Coseno HS entre $\Sigma_c^s$ y $\Sigma_c^t$** | Gram en bloques con congruencia | **Sí**: nula si y solo si $D_{CS,c}=0$ | $O(n^2)$ | Gradiente en $r_c$ | [derivado] |

La lectura para el tutor es esta. **La Rényi matricial y la CS no compiten con los embeddings de Hilbert: son cálculos en el espacio de Hilbert con semántica informacional.** Con $\alpha=2$, la ventaja frente a la MMD no está en «ver más estructura». Está en la invariancia a la escala, la geometría de coseno y, con la normalización r29, una pérdida acotada que identifica la alineación. La ventaja espectral genuina exige $\alpha\ne2$ y pagar $O(n^3)$.

### B.6.3 Esquemas de ponderación

| Esquema | Fórmula por muestra | Cero en la uniforme | ¿Desacoplado? | Uso | Evidencia |
|---|---|---|---|---|---|
| **CREDA (r29 / 2025)** | $1-\hat H_2/\ln C$ | Sí | **No** (r29); no declarado (2025) | Alineación (compuerta) | propuesta; [paper] |
| CDAN+E | $1+e^{-\hat H_1}$ | No (suelo $1+1/C$) | GRL (oficial) / sí (TLlib) | Alineación adversaria | [code](https://raw.githubusercontent.com/thuml/CDAN/master/pytorch/loss.py) |
| SRADA | Creciente en $H$ | — | UNVERIFIED | Alineación adversaria | [arXiv](https://arxiv.org/abs/2006.00223) |
| SHOT (calculado, sin uso) | $1-\hat H_1/\ln C$ | Sí | `no_grad` | Ninguno | [code](https://raw.githubusercontent.com/tim-learn/SHOT/master/object/image_target.py) |
| PLUE | $e^{-\hat H_1/\ln C}$ | No (suelo $e^{-1}$) | Sí | CE | [code](https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation/blob/main/train_target.py) |
| EATA | $e^{E_0-H}$ tras filtro | Filtrado | Sí | Entropía TTA | [code](https://github.com/mr-eggplant/EATA/blob/main/eata.py) |
| CoWA | LPG·MPPL | Casi | Sí | CE/KL con *mixup* | [code](https://github.com/Jhyun17/CoWA-JMDS/blob/main/image_target_CoWA.py) |
| ATDOC-NA | Máximo del agregado de vecinos | No | Sí (peso y denominador) | CE ponderada | [code](https://raw.githubusercontent.com/tim-learn/ATDOC/main/demo_uda.py) |
| LMMD / CPDA | $g_{jc}/\sum g$ | No | Sí | MMD por clase | [code](https://raw.githubusercontent.com/jindongwang/transferlearning/master/code/DeepDA/loss_funcs/lmmd.py) |
| RWOT | $Q(j,c)$ | No | No declarado | Coste de transporte | [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Reliable_Weighted_Optimal_Transport_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf) |
| CAN / DCAN / TPN / DaC / PFAN | Binario ($D_0$; 0,95; 0,6; 0,97; $\tau(m)$) | Sí | Sin gradiente | Alineación o CE | [paper](https://arxiv.org/pdf/1901.00976) |
| SoftMatch | Gaussiana truncada en $\mu_t$ (EMA) | No | Sí | CE | [code](https://raw.githubusercontent.com/microsoft/Semi-supervised-learning/main/semilearn/algorithms/softmatch/utils.py) |

**Algebra del peso de CREDA [derivado].** $w=\log_C(C\sum_kg_k^2)=1+\log_C(1-\mathrm{Gini}(g))$. Como $\hat H_2\le\hat H_1$, $w$ es mayor o igual que la certeza de Shannon normalizada que calcula SHOT. Como $\log_Cx\ge(x-1)/(C-1)$ en $[1,C]$, también es mayor o igual que la certeza lineal de Gini/Tsallis-2. **Es la más permisiva de las tres formas con cero en la uniforme**, y la única forma Rényi-2 encontrada en el cribado. Dentro de $\Sigma_c^t$ entra al cuadrado, lo que la vuelve más estricta allí.

### B.6.4 Justificación de α=2

El PDF de CREDA da su razón [paper]: con kernel gaussiano, el objetivo de orden 2 se convierte en «una suma de similitudes por pares» (el potencial de información) que favorece «clústeres compactos y puros», con sensibilidad a estadísticos de orden superior y consistencia heredada de la estimación por kernel. **No reporta ninguna ablación de $\alpha$** ([PDF CREDA, p. 9](https://mdpi-res.com/d_attachment/mathematics/mathematics-13-02602/article_deploy/mathematics-13-02602.pdf)). La literatura no aporta optimalidad. La única ablación en la línea matricial, de selección de características con $\alpha\in\{0{,}6;1{,}01;2\}$, encontró que $\alpha=2$ «se degradó significativamente en la mayoría de los conjuntos». La heurística de Yu et al. llama a $\alpha=2$ «ponderación neutra» ([ar5iv 1808.07912](https://ar5iv.labs.arxiv.org/html/1808.07912)) [verificado]. En DA con la divergencia $\alpha$ de Amari, que es otro objeto, IT-RUDA encontró un óptimo plano en $\alpha\approx0{,}6$–$0{,}95$ ([arXiv 2210.12947](https://arxiv.org/html/2210.12947)) [verificado].

La justificación defendible tiene tres partes y no incluye optimalidad. La primera es **exactitud computacional**: $H_2(A)=-\ln\|A\|_F^2$ no necesita descomposición espectral, cuesta $O(n^2)$ y su gradiente $\propto A^{\alpha-1}/\mathrm{tr}(A^\alpha)$ es **lineal** en $A$ con $\alpha=2$ ([DIB, arXiv 2102.00533](https://arxiv.org/pdf/2102.00533v1.pdf)) [verificado]. Con $\alpha<1$, en cambio, los autovalores pequeños aparecen elevados a una potencia negativa en el gradiente, lo que puede explotar con Gram casi singulares, por ejemplo con embeddings duplicados [derivado]. La segunda es **estructural, y propia de esta tesis**: solo con $\alpha=2$ la pérdida r29 admite la forma cerrada en $D_{CS,c}$ que da identificabilidad [derivado]. La tercera es la heurística de ponderación neutra. El orden del argumento debe invertirse: $\alpha=2$ se elige por exactitud y estructura, y su optimalidad se **mide** con un barrido sintético. No se encontró ningún precedente de ese barrido para alineación con Rényi matricial. El diseño cruzaría $\alpha\in\{0{,}5;0{,}8;1{,}01;1{,}5;2;3;5\}$ con $\sigma$ y con cuatro desplazamientos controlados: rotación de clase, traslación de una sola clase, escalado anisótropo (que prueba directamente $r_c$) y desplazamiento de proporciones (que prueba $q_c$). Las métricas serían la potencia de una prueba de dos muestras, como en Yin et al. ([arXiv 2405.19978](https://arxiv.org/html/2405.19978v1)), la exactitud tras adaptar y la tasa de NaN y las normas de gradiente.

### B.6.5 Dos ideas de novedad

**Idea 1: divergencia de Cauchy–Schwarz contrastiva y desacoplada de la pureza, con prueba de dos muestras.** La identidad $U_c-J_c=\ln\frac{1+r_c}{r_c+e^{-D_{CS,c}}}$ indica qué separar. Se propone tomar como objeto principal $D_{CS,c}=-\ln\cos_{HS}(\Sigma_c^s,\Sigma_c^t)$, tratar $r_c$ como diagnóstico desacoplado o penalizado de forma explícita (lo que elimina el atajo del gradiente) y añadir el término que le falta frente a CDD y DARES, una repulsión de segundo orden:
$$\mathcal L^{CS}_{glob}=\sum_c\omega_c\,D_{CS}(\Sigma_c^s,\Sigma_c^t)-\eta\sum_{c\ne c'}\omega_c\,\min\big(m,\ D_{CS}(\Sigma_c^t,\Sigma_{c'}^s)\big)\quad\text{[derivado]}.$$
La afirmación teórica que el capítulo podría reclamar es de **identificabilidad**. Si $(\kappa^I_\sigma)^2$ es característico, entonces $D_{CS,c}=0$ si y solo si las distribuciones condicionales de los embeddings de la clase $c$ coinciden entre dominios. Es la versión condicional del Teorema 6 de RJSD ([arXiv 2305.16446](https://arxiv.org/abs/2305.16446)). Falta la demostración formal. La segunda pieza cubre la carencia más visible frente a la MMD: **una prueba por permutación de muestras dentro de cada clase pseudo-etiquetada**, cuya potencia se mediría con el diseño de B.6.4. Así la medida pasaría a **detectar** desplazamiento condicional por clase, algo que ningún trabajo revisado ofrece.

**Idea 2: información mutua condicional matricial $I_\alpha(Z;\mathcal D\mid Y)$ sin enrutamiento duro.** Las $C$ puntuaciones enrutadas por $\tilde y$ se sustituirían por una sola medida de independencia condicional entre el embedding y la variable de dominio dada la clase, construida con productos de Hadamard:
$$I_\alpha(Z;\mathcal D\mid Y)=S_\alpha(Z,Y)+S_\alpha(\mathcal D,Y)-S_\alpha(Z,\mathcal D,Y)-S_\alpha(Y),$$
con una Gram de etiquetas blanda $G_{jj'}=\langle g_j,g_{j'}\rangle$ en el objetivo y one-hot en el origen ([arXiv 1211.2459](https://arxiv.org/abs/1211.2459); [arXiv 1808.07912](https://ar5iv.labs.arxiv.org/html/1808.07912)) [verificado para las definiciones]. Conceptualmente une MIDA (HSIC con la variable de dominio, sin clase) con la condicionalidad de JDA y CREDA, y **elimina de raíz los dos defectos abiertos**. No hay peso que desacoplar, porque la confianza entra como estructura del kernel de etiquetas. No hay clases vacías que excluir. Y el enrutamiento blando de LMMD queda incorporado. Los límites deben declararse: con $\alpha=2$, la información mutua matricial omite el término cruzado del HSIC [derivado], y los operadores de covarianza capturan solo en parte la independencia condicional ([arXiv 2202.08545](https://arxiv.org/abs/2202.08545)) [UNVERIFIED: fragmento]. Por eso esta idea es el lugar natural del barrido de $\alpha$, con $\alpha\to1$ como candidato teóricamente preferible.

## B.7 Huecos que quedan en la comprobación de novedad

La afirmación de novedad descansa en una **ausencia de evidencia tras búsquedas dirigidas, no en una prueba de ausencia**. Hay tres huecos concretos.

**Pasada en Google Scholar.** No se ejecutó. Falta la consulta «cited by» de CREDA. Semantic Scholar registra una sola cita de CREDA (Dardeno et al., de pasada) y OpenAlex cero, pero las dos bases pueden perder citas en revistas regionales o en español ([S2 API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/math13162602)) [verificado].

**Cauchy–Schwarz DA como pariente teórico-informacional más cercano.** La distinción formal que separa a CREDA es $p(\Phi\mid y)$ con operadores de covarianza frente a $p(\hat y\mid z)$ con embeddings de media. Se apoya en el texto de Yin et al. ([arXiv 2405.19978](https://arxiv.org/pdf/2405.19978)), pero no se releyó Yu et al. 2023, la fuente de la CS condicional que CS-DA cita. Tampoco se encontró ningún artículo que enuncie una divergencia CS entre operadores de covarianza de kernel para DA, así que la identidad de B.2 sigue siendo propia.

**Lecturas pendientes en el propio embudo.** Falta comprobar F3 en CAT, CGDM, InAIt, SymNets, DADA y NOUN. Falta saber si el código de RWOT, TPN, DCAN y CAN desacopla. Y hay que contrastar con su PDF todas las ecuaciones marcadas [memory] antes de llevarlas al manuscrito, incluidas casi todas las adversarias, salvo el código de CDAN.

---

## Conclusión

El embudo invierte la pregunta que el capítulo debería responder. «¿Es CREDA condicional y consciente de la confianza?» tiene 23 respuestas afirmativas en la literatura, así que no puede ser el argumento. El argumento es geométrico: CREDA con dos codificadores es, entre todos los métodos cribados, el único que compara **operadores de covarianza de clase** con una distancia de coseno invariante a la escala, frente a una referencia de origen congelada, con una pérdida acotada que se anula exactamente en la alineación. Dos de esas propiedades se pueden demostrar en el papel hoy mismo: la identidad en $D_{CS}$ y el que la mezcla por congruencia sea semidefinida positiva, cosa que la mezcla de la versión publicada de CREDA no garantiza. La otra, el origen congelado, es una decisión de diseño que solo una ablación puede validar frente a ADDA, LMMD y CDD montados sobre los mismos dos codificadores.

El orden de trabajo que se desprende es concreto. Primero, cerrar los dos atajos del gradiente, desacoplando $w$ y $r_c$, porque mientras existan cualquier resultado puede deberse a un colapso premiado y no a la geometría. Segundo, decidir si se incorpora una repulsión interclase en forma CS, que es lo que la Idea 1 propone y lo que CDD y DARES ya tienen. Tercero, completar la búsqueda de citas de CREDA antes de escribir la frase de novedad. Con esas tres piezas resueltas, la tesis podrá defender una afirmación más estrecha que «primer método de este tipo», pero que resista una revisión.
