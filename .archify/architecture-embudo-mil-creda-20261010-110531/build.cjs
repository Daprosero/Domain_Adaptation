const fs=require('fs');
const tr=JSON.parse(fs.readFileSync('/Users/diego/.claude/skills/archify/examples/locales/es.json','utf8'));
const translations=tr.translations||tr.messages||tr;
const C=(id,type,label,sublabel,tag,pos,size)=>({id,type,label,sublabel,tag,pos,size});
const fam=(i,y)=>[40,y];
const doc={schema_version:1,diagram_type:"architecture",
meta:{title:"Del embudo SOTA a las ideas de MIL-CREDA",output:".archify/architecture-embudo-mil-creda-20261010-110531/embudo-mil-creda.html",quality_profile:"showcase",locale:"es",translations,
legend:{entries:{external:{label:"Embudo de revisión"},backend:{label:"Familia superviviente"},security:{label:"Filtro"},frontend:{label:"MIL-CREDA (nuestro)"},database:{label:"Idea candidata"}}}},
components:[
C("embudo","external","Embudo SOTA · 164 métodos","F1 condicional 81 → F2 sin etiquetas 78 → F3 confianza 28","16 con mecanismo verificado",[40,40],[320,72]),
C("fam1","backend","F1 · MMD condicional con filtrado","‖μ̂_c^s − μ̂_c^t‖²_ℋ, kernel característico + umbral","falla: solo nivel instancia",fam(1,170),[320,72]),
C("fam2","backend","F2 · Prototipos y autoentrenamiento","Ψ hacia centroides o CE ponderada por confianza","falla: alinea puntos (1.er orden)",fam(2,270),[320,72]),
C("fam3","backend","F3 · Adversarial con peso por entropía","discriminador sobre z⊗g, w = 1 + e^(−H)","falla: min-max implícito, w ≥ 1",fam(3,370),[320,72]),
C("fam4","backend","F4 · Transporte óptimo con confiabilidad","plan de transporte, costo escalado","falla: costo alto, nivel instancia",fam(4,470),[320,72]),
C("fam5","backend","F5 · Informacional condicional","J = ½(H₂ˢ+H₂ᵗ) − H₂(mezcla)","falla: mezcla indefinida",fam(5,570),[320,72]),
C("filtro4","security","Filtro F4 · frente a CREDA","cubiertas: F3, F4 · referencia: F5","pasan: CDD/CAN, CoWA-JMDS, ProDA, DARES",[480,170],[240,472]),
C("filtrofinal","security","Filtro final · frente a MIL-CREDA","todos operan por instancia, i.i.d.","ninguno toma al sujeto como unidad",[860,330],[240,152]),
C("milcreda","frontend","MIL-CREDA r29","J_c = log(½cosh(ΔH/2) + ½e^(−D_CS))","ℓ_c · compuerta q_c · término local μ_j^s",[1240,316],[290,180]),
C("idea1","database","Idea I · Alineación condicional por sujeto","D_CS = 0 ⇔ P_s(B|c) = P_t(B|c)","CANDIDATA — chequeo de novedad en curso",[1680,150],[320,120]),
C("idea2","database","Idea II · Referencia fuente por sujeto","μ_j^s = Σ_i π_ji Ψ(B_i^s), d_j² = ‖Ψ(B_j^t) − μ_j^s‖²","CANDIDATA — chequeo de novedad en curso",[1680,560],[320,120])
],
boundaries:[{kind:"region",label:"Cinco familias supervivientes",wraps:["fam1","fam2","fam3","fam4","fam5"]}],
connections:[
{id:"embudo-familias",from:"embudo",to:"fam1",label:"28 → 5 familias",fromSide:"bottom",toSide:"top"},
{id:"f1-aporta",from:"fam1",to:"filtro4",label:"aporta",variant:"emphasis"},
{id:"f2-aporta",from:"fam2",to:"filtro4",label:"aporta",variant:"emphasis"},
{id:"f3-cubierta",from:"fam3",to:"filtro4",label:"cubierta",variant:"dashed"},
{id:"f4-cubierta",from:"fam4",to:"filtro4",label:"cubierta",variant:"dashed"},
{id:"f5-referencia",from:"fam5",to:"filtro4",label:"referencia"},
{id:"filtro4-final",from:"filtro4",to:"filtrofinal",label:"4 supervivientes"},
{id:"final-milcreda",from:"filtrofinal",to:"milcreda",label:"sin nivel sujeto",variant:"emphasis"},
{id:"milcreda-idea1",from:"milcreda",to:"idea1",label:"vehículo: mezcla, D_CS",variant:"emphasis",fromSide:"right",toSide:"bottom"},
{id:"milcreda-idea2",from:"milcreda",to:"idea2",label:"término local",variant:"emphasis",fromSide:"right",toSide:"top"},
{id:"final-idea1",from:"filtrofinal",to:"idea1",label:"supera: solo identificables por instancia",variant:"dashed",fromSide:"top",toSide:"left",via:[[980,210]]},
{id:"final-idea2",from:"filtrofinal",to:"idea2",label:"supera: sin referencia por sujeto",variant:"dashed",fromSide:"bottom",toSide:"left",via:[[980,620]]}
],
cards:[
{dot:"cyan",title:"Embudo",items:["164 métodos revisados: adversariales; discrepancia/kernel/OT; autoentrenamiento/SFDA/destilación; transformers/TTA; EEG/BCI/MIL","F1 condicional: 81 → F2 sin etiquetas del objetivo: 78 → F3 ponderación por confianza: 28 (16 verificados)"]},
{dot:"emerald",title:"Familia 1 · MMD condicional con filtrado",items:["Miembros: CAN/CDD, DCAN/CMMD (laxos: LMMD, CPDA, PCT)","Hace: ‖μ̂_c^s − μ̂_c^t‖²_ℋ con kernel característico; filtra muestras dudosas por umbral","Falla: identificable solo a nivel de instancia; umbrales fijos; MMD no normalizada","Fortaleza honesta: kernel característico; el nuestro es lineal sobre bolsas y solo ve segundo orden"]},
{dot:"emerald",title:"Familia 2 · Prototipos y autoentrenamiento",items:["Miembros: TPN, PFAN, ProDA, ATDOC-NA, SHOT++, DaC, A2Net, PLUE, T3A, CoWA-JMDS","Hace: acerca Ψ a centroides de clase o usa CE con pseudoetiqueta ponderada por confianza","Falla: alinea puntos (primer orden), no distribuciones; CoWA usa covarianzas por clase solo en sus pesos"]},
{dot:"emerald",title:"Familia 3 · Adversarial condicionado",items:["Miembros: CDAN+E, SRADA","Hace: discriminador sobre z⊗g, w = 1 + e^(−H)","Falla: divergencia implícita (min-max), sin forma cerrada, inestable; w ≥ 1 nunca excluye una muestra"]},
{dot:"emerald",title:"Familia 4 · Transporte óptimo con confiabilidad",items:["Miembros: RWOT","Hace: plan de transporte con costo escalado por confiabilidad","Falla: costo alto, nivel de instancia, sin estructura de clase en la medida"]},
{dot:"emerald",title:"Familia 5 · Informacional condicional",items:["Miembros: CREDA, DARES","Hace: J = ½(H₂ˢ+H₂ᵗ) − H₂(mezcla), mezcla desbalanceada, bloque cruzado sin ponderar","Falla: la mezcla puede ser indefinida (autovalor −0.131 con k = 0.5, w = 0.3); el máximo de J no es alineación; nivel de instancia; un codificador"]},
{dot:"rose",title:"Filtro F4 · frente a CREDA (precursor, nivel instancia)",items:["Familia 5 (CREDA/DARES) es la referencia","Cubiertas por CREDA: Familia 3 (divergencia explícita en forma cerrada en lugar de min-max) y Familia 4 (O(n²), sin plan de transporte)","Aportan lo que CREDA no tiene: Familia 1 (CDD: kernel característico + repulsión entre clases), Familia 2 (CoWA: covarianzas por clase en sus pesos; ProDA: referencia prototípica suave) y DARES (repulsión, piso anticolapso)","Supervivientes: CDD/CAN, CoWA-JMDS, ProDA, DARES"]},
{dot:"rose",title:"Filtro final · frente a MIL-CREDA r29",items:["Todos caen en el mismo eje: operan por instancia, i.i.d.","Ninguno toma al sujeto (bolsa) como unidad, ninguno alinea distribuciones condicionales de sujetos, ninguno construye una referencia personalizada por sujeto","Conservan algo que MIL-CREDA debería adoptar o ablacionar: la repulsión de CDD, los pesos con covarianza de CoWA, los pesos desacoplados (detach)"]},
{dot:"violet",title:"MIL-CREDA r29 · nuestro",items:["J_c = log(½cosh(ΔH/2) + ½e^(−D_CS)): el término global es una divergencia de Cauchy–Schwarz entre operadores de covarianza por clase","ℓ_c = (U_c−J_c)/(U_c−L_c); ℓ_c = 0 ⇔ D_CS = 0","Bolsas como embeddings medios; mezcla de congruencia balanceada (siempre PSD); compuerta q_c; referencia personalizada μ_j^s; dos codificadores"]},
{dot:"violet",title:"MIL-CREDA r29 · heredado de CREDA",items:["w = 1 − H₂/ln C","Enrutamiento duro por pseudoetiqueta","Forma de J_c"]},
{dot:"amber",title:"MIL-CREDA r29 · pendiente",items:["Desacoplar (detach) w","Desacoplar r en el numerador","Repulsión entre clases (como en CDD/DARES)"]},
{dot:"orange",title:"Idea I · Alineación condicional identificable a nivel de sujeto (CANDIDATA — chequeo de novedad en curso)",items:["Cada sujeto es una bolsa, una medida sobre sus instancias","Kernel característico de dos niveles κ(B,B') = exp(−‖Ψ(B)−Ψ(B')‖²/2τ²) sobre los embeddings medios Ψ","D_CS = 0 ⇔ P_s(B|c) = P_t(B|c): el término condicional se anula si y solo si coinciden las distribuciones condicionales de SUJETOS entre dominios","Vehículo: la maquinaria de r29 (mezcla balanceada, ℓ_c = 0 ⇔ D_CS = 0)","Supera a los supervivientes de F4: solo son identificables sobre instancias"]},
{dot:"orange",title:"Idea II · Referencia fuente personalizada por sujeto (CANDIDATA — chequeo de novedad en curso)",items:["El término global enruta por pseudoetiqueta dura (heredado de CREDA)","El término local construye, para cada sujeto objetivo, μ_j^s = Σ_i π_ji Ψ(B_i^s): baricentro en ℋ_I ponderado por la posterior COMPLETA g_j^t","Penaliza d_j² = ‖Ψ(B_j^t) − μ_j^s‖²: un sujeto mal enrutado sigue atraído hacia la mezcla de las clases entre las que realmente duda","Alineación a dos escalas: condicional por clase (global, dura) + personalizada por sujeto (local, suave)","Supera a ProDA (prototipos por instancia, no baricentro de sujeto) y a CDD/CoWA (sin referencia por sujeto)"]}
]};
fs.writeFileSync('candidate.json',JSON.stringify(doc,null,1));
