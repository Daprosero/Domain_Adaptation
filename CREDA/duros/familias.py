#!/usr/bin/env python3
"""Three harder toy families, all in 3D, with the classes carved out of the structure.

The families in use arrive at the encoder already flattened -- 1.11 and 1.04
effective dimensions of an ambient 2 or 3 -- because they take 3 of 10 labels out
of a generator whose labels do not follow its geometry. These three are built the
other way round: the class IS a piece of the shape, so the three classes cannot be
collinear unless the shape is.

Ordered easiest to hardest, which is the order they are drawn in:

  lunas     two opposite parabolas, extruded into sheets. One sheet carries two
            classes split along its length, the other carries the third. Hard
            because two of the three classes are adjacent on the SAME curve.

  toroide   a full torus cut into three 120-degree arcs of the major angle. The
            three classes sit on a ring, so no one of them is between the other
            two -- the geometry the kernel needs, by construction.

  rollo     a full Swiss roll cut into three consecutive segments along its
            parameter: stretch it, cut in three, roll it back up. The hardest,
            because the segments interleave in space while being far apart along
            the manifold.

The perturbation is scaled to each family's OWN spread rather than fixed, so
'does not deform the structure much' is a ratio and not an impression: the noise
is a percentage of the data's own standard deviation, and the rotation is a real
3D rotation about a generic axis rather than an axis-aligned one.
"""
import numpy as np

RUIDO_REL   = 0.025      # sigma del ruido, como fraccion de la dispersion propia
TRASLADO_REL= 3.0        # traslacion, en unidades de la dispersion propia
ANGULO      = np.pi/6    # 30 grados, alrededor de un eje generico

def _rotacion_generica(angulo, eje=(1.0, 0.7, 0.4)):
    """Rodrigues: una rotacion de verdad en 3D, no una alineada a un eje.

    Una rotacion alineada a un eje deja una coordenada intacta, y entonces el
    problema que plantea es mas facil de lo que su angulo sugiere.
    """
    k = np.array(eje, dtype=float); k = k/np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angulo)*K + (1 - np.cos(angulo))*(K @ K)

def perturbar(X, rng):
    """Rotacion generica + traslacion + ruido, todo a escala de los propios datos."""
    esc = float(X.std(0).mean())
    R = _rotacion_generica(ANGULO)
    d = np.full(3, TRASLADO_REL*esc/np.sqrt(3))
    return (X - X.mean(0)) @ R.T + X.mean(0) + d + rng.normal(0, RUIDO_REL*esc, X.shape)

# ---------------------------------------------------------------- las familias
def lunas(n, rng):
    """Dos parabolas opuestas, extruidas en laminas. 3 clases: dos en una lamina.

    La lamina de arriba lleva DOS clases partidas por la mitad de su largo y la
    de abajo lleva la tercera, asi que dos de las tres son vecinas sobre la MISMA
    curva. Las tres se muestrean con n/3 cada una: un desbalance 1:1:2 seria un
    confundidor, porque la clase mayoritaria arrastra la media global.
    """
    k = n//3
    # El brazo llega a 1.6 y la extrusion a 0.5: la parabola tiene que dominar la
    # lamina, o la figura se lee como dos planos y no como dos curvas opuestas.
    BRAZO, ALTO = 1.6, 0.5
    t1 = rng.uniform(-BRAZO, 0.0, k)        # clase 0: mitad izquierda de la lamina A
    t2 = rng.uniform(0.0, BRAZO, k)         # clase 1: mitad derecha de la misma
    t3 = rng.uniform(-BRAZO, BRAZO, n - 2*k)  # clase 2: la lamina B entera
    z = lambda m: rng.uniform(-ALTO, ALTO, m)
    X = np.vstack([np.column_stack([t1, t1**2, z(k)]),
                   np.column_stack([t2, t2**2, z(k)]),
                   np.column_stack([t3 + 1.5, -t3**2 + 1.4, z(n - 2*k)])])
    y = np.concatenate([np.zeros(k), np.ones(k), np.full(n - 2*k, 2)])
    return X, y.astype(np.int64)

def toroide(n, rng):
    """Un toro completo, cortado en tres arcos de 120 grados del angulo mayor."""
    R, r = 3.0, 1.1
    phi = rng.uniform(0, 2*np.pi, n)
    th  = rng.uniform(0, 2*np.pi, n)
    X = np.column_stack([(R + r*np.cos(th))*np.cos(phi),
                         (R + r*np.cos(th))*np.sin(phi),
                         r*np.sin(th)])
    y = np.floor(phi/(2*np.pi/3)).astype(np.int64) % 3
    return X, y

def rollo(n, rng):
    """Un rollo suizo completo: estirarlo, cortarlo en tres y volver a enrollarlo."""
    t = rng.uniform(1.5*np.pi, 4.5*np.pi, n)
    h = rng.uniform(0, 12.0, n)
    X = np.column_stack([t*np.cos(t), h, t*np.sin(t)])
    cortes = np.linspace(1.5*np.pi, 4.5*np.pi, 4)
    y = (np.digitize(t, cortes[1:-1])).astype(np.int64)
    return X, y

FAMILIAS = [('lunas', lunas), ('toroide', toroide), ('rollo', rollo)]

def construir(nombre, fn, n=1800, semilla=0):
    rng = np.random.default_rng(semilla)
    Xs, ys = fn(n, rng)
    Xt, yt = fn(n, np.random.default_rng(semilla + 1000))
    return Xs, ys, perturbar(Xt, rng), yt

def dim_ef(X):
    w = np.clip(np.linalg.eigvalsh(np.cov(X.T)), 1e-15, None)
    return float(w.sum()**2/(w**2).sum())

def entre_dentro(X, y):
    mu = X.mean(0); e = d = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        e += len(Xc)*float(((Xc.mean(0)-mu)**2).sum())/len(X)
        d += float(((Xc-Xc.mean(0))**2).sum())/len(X)
    return 100*e/(e+d)
