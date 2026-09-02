"""La negativa a agrupar una dimensión que la declaración llama `perRun`.

Vivía dentro de `tables`, y ahí alcanzaba a la familia del veredicto: todo lo
que sale de las corridas crudas pasa por `cells`, y la guarda estaba en `cells`.
La familia del ruido no pasa por ahí --- agrega por `contamination.by_arm` ---
así que la misma regla no la tocaba, y `conclusion_versus_clean("seconds", ρ)`
promediaba tiempo de pared sobre transferencias, repeticiones y máquinas antes
de imprimir quién pierde menos.

Está en un módulo propio y no en `tables` por la dirección: `contamination` es
una junta de datos y `figures` dibuja; ninguno de los dos tiene por qué importar
al que formatea tablas para conseguir una regla que es de la declaración. Y sin
un lugar común la salida era copiar el texto, que es de lo que trata este
archivo: cinco copias del mismo párrafo son cinco textos que pueden separarse, y
el que se quede viejo va a sonar igual de firme.

Liviano a propósito: `shards` se importa adentro de la función porque arrastra
`harness`, y con él `torch`, a módulos que hoy sólo leen JSON o formatean.
"""

from __future__ import annotations


def per_run_dimensions() -> list[str]:
    """Las dimensiones que la declaración del propio banco dice que no se agrupan.

    Se lee de la declaración y no se lista acá: quién es `perRun` lo decide el
    repositorio que mide, no el módulo que imprime. Copiar la lista sería una
    segunda fuente de verdad que envejece en silencio.
    """
    from MIL_CREDA_Benchmark import shards
    return list(shards.declaration().get("perRun") or [])


def refuse(metric: str) -> None:
    """La negativa, escrita una vez, para toda función que agrupe lo que le pasan.

    Se llama en dos clases de sitio y las dos hacen falta. En el punto donde de
    verdad se promedia ---`cells`, `by_arm`--- para que la próxima función que
    lea corridas nazca cubierta. Y en la entrada de cada función pública que
    puede volver temprano: `curve` sin registros no llega a `by_arm`, y
    `conclusion_versus_clean` corta en «falta el registro» antes de agregar
    nada. Con la guarda sólo en el punto de agregación, un árbol vacío las deja
    pasar a todas y la suite queda verde sin que nada se haya negado --- y la
    negativa aparecería recién el día que hay campaña, que es el día en que ya
    se imprimió.
    """
    prohibidas = per_run_dimensions()
    if metric not in prohibidas:
        return
    raise ValueError(
        f"refusing to pool `{metric}`: the benchmark declares it `perRun` "
        f"({', '.join(prohibidas)}).\n"
        "  No reading of it was stable enough to stand for the method, or "
        "even for one machine across two of its own runs, so a "
        "`mean ± stdev` here would describe none of the runs behind it.\n"
        "  Use `render_per_run(summary['gridPerRun'], ...)`, which prints "
        "every reading with the run that produced it."
    )
