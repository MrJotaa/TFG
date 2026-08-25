export const CLASSIFICATION_METRIC_GUIDES = {
    overview: {
        title: 'Cómo leer el resumen de clasificación',
        summary:
            'Distingue el rendimiento antes y después del ataque y cuantifica cuánto cambia.',
        items: [
            {
                term: 'Exactitud original y robusta',
                definition:
                    'Porcentaje de imágenes clasificadas correctamente antes y después del ataque. Un valor mayor es mejor.',
            },
            {
                term: 'Caída y ASR condicional',
                definition:
                    'La caída es original menos tras el ataque, en puntos porcentuales. El ASR condicional mide qué proporción de las predicciones originalmente correctas pasa a ser incorrecta.',
            },
            {
                term: 'Precisión, recall y F1',
                definition:
                    'La precisión penaliza falsos positivos, el recall penaliza casos reales no recuperados y F1 equilibra ambas. “Macro” da el mismo peso a cada clase; “ponderado” usa su número de muestras.',
            },
            {
                term: 'Probabilidad y perturbación',
                definition:
                    'La caída de probabilidad usa la clase real. L∞ es el mayor cambio en un píxel y L2 resume la magnitud total de la perturbación.',
            },
        ],
        note: 'Salvo que se indique lo contrario, los deltas se calculan como original − tras el ataque: un valor positivo representa degradación.',
    },
    classes: {
        title: 'Cómo leer las métricas por clase',
        summary:
            'Compara qué categorías conservan el rendimiento y cuáles concentran la degradación.',
        items: [
            {
                term: 'Soporte',
                definition:
                    'Número de muestras reales de la clase. Con soporte bajo, una sola imagen puede cambiar mucho el resultado.',
            },
            {
                term: 'Precisión, recall y F1',
                definition:
                    'Se calculan considerando una clase frente al resto. F1 solo es alto cuando la precisión y el recall son altos.',
            },
            {
                term: 'ROC-AUC y AP (Average Precision)',
                definition:
                    'ROC-AUC mide la capacidad de ordenar positivos por encima de negativos. AP (Average Precision) resume la curva Precision–Recall y suele ser más informativa con clases poco frecuentes.',
            },
            {
                term: 'Degradación',
                definition:
                    'Es el valor original menos el obtenido tras el ataque, expresado en puntos porcentuales. Un valor negativo indica una mejora en esa ejecución, no mayor degradación.',
            },
        ],
        note: 'Una métrica puede no estar definida cuando la clase no contiene suficientes positivos o negativos para calcularla.',
    },
    matrices: {
        title: 'Cómo leer las matrices de confusión',
        summary:
            'Localiza qué clases se confunden entre sí antes y después del ataque.',
        items: [
            {
                term: 'Filas, columnas y diagonal',
                definition:
                    'Cada fila es la clase real y cada columna la predicha. La diagonal contiene aciertos; las celdas fuera de ella muestran confusiones.',
            },
            {
                term: 'Conteos',
                definition:
                    'Muestran el número absoluto de imágenes de cada combinación real-predicha y permiten valorar cuántos casos representa una confusión.',
            },
            {
                term: 'Normalizada por clase real',
                definition:
                    'Cada fila se expresa como porcentaje de su clase real. Facilita comparar patrones entre clases con distinto soporte.',
            },
            {
                term: 'Vista compacta y completa',
                definition:
                    'La compacta oculta filas sin muestras reales, pero conserva las columnas de clases que el modelo sí predijo. La completa muestra todo el catálogo.',
            },
        ],
        note: 'Compara las matrices original y tras el ataque: un desplazamiento de masa desde la diagonal hacia otras columnas revela nuevas confusiones.',
    },
    curves: {
        title: 'Cómo leer las curvas de clasificación',
        summary:
            'Observa el comportamiento al variar el umbral de decisión, no solo en un umbral fijo.',
        items: [
            {
                term: 'Ámbito micro global',
                definition:
                    'El promedio micro agrupa todas las decisiones clase-frente-al-resto de todas las imágenes. Las clases con más decisiones influyen más en la curva.',
            },
            {
                term: 'Ámbito de una clase',
                definition:
                    'Trata esa clase como positiva y todas las demás como negativas. El soporte “n” indica cuántos positivos reales contiene.',
            },
            {
                term: 'Curva ROC y ROC-AUC',
                definition:
                    'ROC enfrenta verdaderos positivos con falsos positivos. El área bajo la curva va de 0 a 1; cuanto mayor, mejor separación.',
            },
            {
                term: 'Precision–Recall (PR) y AP',
                definition:
                    'La curva enfrenta precisión con recall. AP (Average Precision) resume su comportamiento: cuanto mayor, mejor equilibrio a través de los umbrales.',
            },
            {
                term: 'Promedio macro',
                definition:
                    'Es el promedio no ponderado de la puntuación de cada clase disponible. Todas las clases aportan el mismo peso.',
            },
        ],
        note: 'La degradación es la puntuación original menos la obtenida tras el ataque. Las curvas pueden no estar definidas si faltan positivos o negativos.',
    },
    images: {
        title: 'Cómo leer los resultados por imagen',
        summary:
            'Relaciona las métricas globales con casos concretos y revisa cómo cambian la predicción y la explicación visual.',
        items: [
            {
                term: 'Transición de predicción',
                definition:
                    'Compara la corrección antes y después del ataque: una transición correcta → incorrecta es un éxito adversarial; incorrecta → correcta es una recuperación.',
            },
            {
                term: 'Confianza, probabilidad y margen',
                definition:
                    'La confianza corresponde a la clase predicha. La probabilidad de la clase real sigue esa etiqueta concreta. El margen es la diferencia entre las dos clases con mayor puntuación.',
            },
            {
                term: 'Grad-CAM',
                definition:
                    'Resalta regiones influyentes para la decisión del modelo. Ayuda a interpretar el cambio, pero no demuestra causalidad por sí sola.',
            },
            {
                term: 'Perturbación L∞ y L2',
                definition:
                    'L∞ recoge el cambio máximo de un píxel; L2 agrega la energía de todos los cambios. No son porcentajes ni se comparan directamente entre sí.',
            },
        ],
        note: 'Los filtros actúan sobre todas las imágenes evaluadas; la paginación solo limita cuántas se muestran a la vez.',
    },
}

export const DETECTION_METRIC_GUIDES = {
    overview: {
        title: 'Cómo leer el resumen de detección',
        summary:
            'Resume la calidad COCO, el punto de operación y las transiciones de los objetos anotados.',
        items: [
            {
                term: 'mAP original y tras el ataque',
                definition:
                    'mAP (mean Average Precision) promedia AP entre clases y umbrales IoU de 0,50 a 0,95. Mide la calidad global antes y después del ataque; un valor mayor es mejor.',
            },
            {
                term: 'Degradación de mAP',
                definition:
                    'Es mAP original menos mAP tras el ataque, en puntos porcentuales. Positivo indica pérdida; negativo, una mejora en esa ejecución.',
            },
            {
                term: 'F1 a punto de operación',
                definition:
                    'Equilibra precisión y recall usando exactamente los umbrales de confianza e IoU elegidos. No es el mismo cálculo que el mAP COCO.',
            },
            {
                term: 'Supresión y otras transiciones',
                definition:
                    'Suprimido significa detectado originalmente y omitido tras el ataque. Preservado, recuperado y no detectado en ambas completan el universo de objetos evaluables.',
            },
            {
                term: 'Indicadores heurísticos históricos',
                definition:
                    'Describen conteos y confianza de las predicciones publicadas. Son orientativos y no sustituyen las métricas supervisadas COCO.',
            },
        ],
        note: 'Original y tras el ataque se calculan sobre las mismas imágenes y anotaciones. Revisa la cobertura antes de generalizar el resultado.',
    },
    coco: {
        title: 'Cómo leer las métricas COCO',
        summary:
            'Compara precisión y recuperación con el protocolo estándar de detección de objetos.',
        items: [
            {
                term: 'mAP@[0.50:0.95]',
                definition:
                    'Promedia AP entre clases y diez umbrales IoU de 0,50 a 0,95. Exige tanto clasificar como localizar correctamente.',
            },
            {
                term: 'AP@0.50 y AP@0.75',
                definition:
                    'AP@0.50 acepta una coincidencia espacial más flexible; AP@0.75 exige cajas más ajustadas. AP resume la curva Precision–Recall.',
            },
            {
                term: 'AP/AR por tamaño',
                definition:
                    'Pequeño, mediano y grande siguen los rangos de área COCO. Permiten detectar si el ataque afecta de forma distinta según el tamaño del objeto.',
            },
            {
                term: 'AR@1, AR@10 y AR@100',
                definition:
                    'AR (Average Recall) indica qué fracción de los objetos de referencia puede recuperarse permitiendo como máximo 1, 10 o 100 detecciones por imagen.',
            },
            {
                term: 'Delta absoluto y relativo',
                definition:
                    'El absoluto es original − tras el ataque en puntos porcentuales. El relativo expresa esa pérdida respecto al valor original; no son la misma unidad.',
            },
            {
                term: 'Punto de operación',
                definition:
                    'TP, FP, FN, precisión, recall y F1 usan los umbrales concretos configurados. COCO AP/AR, en cambio, integra múltiples puntuaciones de confianza e IoU.',
            },
        ],
        note: 'Un valor alto de detecciones no implica mejor calidad: puede incluir falsos positivos. Para interpretar calidad usa AP, AR y el punto de operación.',
    },
    classes: {
        title: 'Cómo leer el desglose por clase',
        summary:
            'Identifica categorías vulnerables sin ocultarlas dentro del promedio global.',
        items: [
            {
                term: 'Soporte',
                definition:
                    'Número de objetos de referencia evaluables de la clase. Con pocos objetos, AP y AR pueden variar mucho ante un solo caso.',
            },
            {
                term: 'AP y AR por clase',
                definition:
                    'AP resume la curva Precision–Recall y AR resume la recuperación de objetos para esa categoría a través de los umbrales COCO.',
            },
            {
                term: 'Δ AP',
                definition:
                    'Es AP original menos AP tras el ataque, en puntos porcentuales. Un valor positivo señala degradación de esa clase.',
            },
            {
                term: 'TP / FP / FN',
                definition:
                    'Son conteos al punto de operación configurado. TP es una detección correcta; FP, una predicción sin correspondencia válida; FN, un objeto omitido.',
            },
            {
                term: 'Transiciones',
                definition:
                    'Preservados, suprimidos y recuperados siguen cada objeto anotado entre la evaluación original y la obtenida tras el ataque.',
            },
        ],
        note: 'Ordenar por degradación ayuda a priorizar clases, pero conviene contrastar siempre el delta con su soporte.',
    },
    curves: {
        title: 'Cómo leer las curvas Precision–Recall de detección',
        summary:
            'Muestra cómo cambia el equilibrio entre detecciones correctas y objetos recuperados al variar la puntuación de confianza.',
        items: [
            {
                term: 'Eje de recall',
                definition:
                    'Fracción de los objetos de referencia recuperada. Avanzar hacia la derecha significa detectar una mayor proporción de objetos reales.',
            },
            {
                term: 'Eje de precisión',
                definition:
                    'Proporción de detecciones publicadas que son correctas. Una curva más alta mantiene menos falsos positivos para el mismo recall.',
            },
            {
                term: 'Promedio COCO global',
                definition:
                    'Promedia la precisión interpolada en 101 niveles de recall, todas las clases evaluables y los IoU de 0,50 a 0,95.',
            },
            {
                term: 'Curva de una clase',
                definition:
                    'Aísla una categoría y conserva el promedio sobre los IoU COCO. “n” indica su número de objetos de referencia.',
            },
            {
                term: 'Área resumida como AP',
                definition:
                    'La media de los puntos de la curva produce AP para una clase y mAP para el promedio global. Una curva superior suele implicar mayor AP.',
            },
        ],
        note: 'Las series original y tras el ataque usan la misma rejilla COCO de recall, por lo que pueden compararse punto a punto.',
    },
    objects: {
        title: 'Cómo leer el análisis por objeto',
        summary:
            'Sigue cada objeto de referencia entre la inferencia original y la obtenida tras el ataque a un punto de operación fijo.',
        items: [
            {
                term: 'Objeto evaluable',
                definition:
                    'Es una anotación de referencia con clase resuelta. Cada objeto aparece una sola vez y se empareja con detecciones de su misma categoría mediante COCOeval.',
            },
            {
                term: 'Preservado, suprimido y recuperado',
                definition:
                    'Preservado se detecta en ambas versiones; suprimido solo en la original; recuperado solo tras el ataque. “No detectado en ambas” no aparece en ninguna.',
            },
            {
                term: 'Confianza e IoU',
                definition:
                    'La confianza es la puntuación de la detección emparejada. IoU mide el solapamiento con la caja de referencia, de 0 a 1. Un valor mayor representa mejor coincidencia.',
            },
            {
                term: 'Δ confianza y Δ IoU',
                definition:
                    'Se calculan como original − tras el ataque y solo existen para objetos preservados. Un delta positivo indica caída tras el ataque.',
            },
            {
                term: 'Filtros, ordenación y páginas',
                definition:
                    'Los filtros y la ordenación se aplican al universo completo de objetos materializados; después se muestran 25, 50 o 100 por página.',
            },
        ],
        note: 'Este análisis usa los umbrales de confianza e IoU configurados. No debe confundirse con TIDE, que fija IoU 0,50.',
    },
    tide: {
        title: 'Cómo leer el diagnóstico TIDE',
        summary:
            'Descompone la pérdida de AP50 según el tipo de error que la provoca.',
        items: [
            {
                term: 'AP50 TIDE',
                definition:
                    'AP (Average Precision) calculada por TIDE con IoU 0,50. Está en escala 0–100 y no sustituye el mAP COCO de 0,50 a 0,95.',
            },
            {
                term: 'dAP',
                definition:
                    'Puntos de AP50 que se recuperarían al corregir hipotéticamente un solo tipo de error. Un dAP mayor implica más impacto, no necesariamente más casos.',
            },
            {
                term: 'Frecuencia e impacto',
                definition:
                    'El conteo indica cuántas veces aparece un error; dAP cuantifica cuánto perjudica al AP50. Un error frecuente puede tener poco impacto y viceversa.',
            },
            {
                term: 'Cambio de impacto',
                definition:
                    'Es dAP tras el ataque menos dAP original. Positivo significa que ese tipo de error pesa más después del ataque.',
            },
            {
                term: 'Oráculos globales FP y FN',
                definition:
                    'Agrupan el impacto recuperable de todos los falsos positivos o falsos negativos, respectivamente.',
            },
        ],
        note: 'Los dAP no deben sumarse: los oráculos interactúan. Un cambio negativo puede indicar que el ataque transformó el error en otro distinto, no que el detector mejoró.',
    },
    images: {
        title: 'Cómo leer los resultados por imagen',
        summary:
            'Contrasta las cajas originales y las obtenidas tras el ataque y genera explicaciones solo para los casos relevantes.',
        items: [
            {
                term: 'Cajas y confianza',
                definition:
                    'Cada caja muestra una clase predicha y su puntuación de confianza. Las vistas original y tras el ataque conservan la geometría real de cada imagen.',
            },
            {
                term: 'Cambios de conteo',
                definition:
                    'Una reducción puede deberse a objetos suprimidos, pero un aumento puede introducir falsos positivos. El conteo aislado no mide calidad.',
            },
            {
                term: 'Indicador heurístico de éxito',
                definition:
                    'Marca una reducción del número de detecciones o de la confianza media. Es descriptivo y no equivale al éxito supervisado medido con anotaciones de referencia.',
            },
            {
                term: 'Eigen-CAM',
                definition:
                    'Visualiza regiones de activación del detector para la imagen original y la obtenida tras el ataque. Ayuda a interpretar el modelo, pero no es una métrica de rendimiento.',
            },
        ],
        note: 'Las cajas mostradas usan el filtro público de confianza > 0,25, que puede diferir del umbral elegido para el punto de operación supervisado.',
    },
}
