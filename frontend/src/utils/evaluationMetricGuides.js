export const CLASSIFICATION_METRIC_HELP = {
    f1_macro: 'Calcula F1 por separado para cada clase y después hace una media en la que todas las clases pesan lo mismo, tengan muchas o pocas muestras.',
    top5_robust_accuracy: 'Porcentaje de imágenes cuya clase real sigue apareciendo entre las cinco predicciones con mayor puntuación después del ataque.',
    conditional_asr: 'Entre las imágenes que el modelo clasificaba correctamente antes del ataque, indica qué porcentaje pasa a clasificarse incorrectamente después.',
    true_class_probability_drop: 'Mide cuánto disminuye, de media, la probabilidad asignada a la clase real después del ataque.',
    linf: 'L∞ es el mayor cambio introducido en un único valor de píxel. Resume la perturbación máxima local.',
    l2: 'L2 combina todos los cambios de la imagen en una sola magnitud. Resume el tamaño global de la perturbación.',
    micro: 'Junta las decisiones clase frente al resto de todas las clases antes de calcular la métrica. Resume el comportamiento global sin forzar que cada clase tenga el mismo peso, como sí hace el promedio macro.',
    macro: 'Calcula la métrica por clase y después hace una media no ponderada: todas las clases aportan el mismo peso.',
    roc_auc: 'Resume la capacidad de separar una clase del resto al variar el umbral. Un valor más próximo a 1 indica mejor separación.',
    average_precision: 'Resume la curva Precision–Recall. Un valor mayor indica un mejor equilibrio entre encontrar positivos y evitar falsos positivos a través de los umbrales.',
}

export const CLASSIFICATION_METRIC_GUIDES = {
    overview: {
        title: 'Cómo leer el resumen de clasificación',
        summary:
            'Empieza comparando el rendimiento original y robusto; después utiliza ASR, probabilidad y perturbación para entender cuánto daño produce el ataque y con qué magnitud.',
        items: [
            {
                term: '1. Compara antes y después',
                definition:
                    'La exactitud original es la referencia del modelo sin ataque. La exactitud robusta muestra cuánto rendimiento conserva después. Cuanto menor sea la separación entre ambas, mayor resistencia presenta el modelo.',
            },
            {
                term: '2. Cuantifica el impacto',
                definition:
                    'La caída de exactitud indica cuántos puntos porcentuales se pierden. El ASR condicional responde a otra pregunta: de las predicciones que eran correctas, cuántas consigue romper el ataque.',
            },
            {
                term: '3. Relaciona daño y perturbación',
                definition:
                    'La caída de probabilidad muestra cuánto pierde confianza el modelo en la clase real. L∞ y L2 describen el tamaño de la perturbación desde dos perspectivas diferentes.',
            },
        ],
        note: 'Salvo que se indique lo contrario, los deltas se calculan como valor original − valor tras el ataque. Un delta positivo suele representar degradación.',
    },
    classes: {
        title: 'Cómo leer las métricas por clase',
        summary:
            'Utiliza esta pestaña para descubrir si el ataque afecta a todas las categorías por igual o si concentra la degradación en clases concretas.',
        items: [
            {
                term: '1. Mira primero el soporte',
                definition:
                    'El soporte es el número de muestras reales de la clase. Con muy pocas muestras, una sola imagen puede provocar cambios porcentuales grandes y la comparación debe interpretarse con cautela.',
            },
            {
                term: '2. Compara la misma métrica',
                definition:
                    'Las barras original y tras el ataque permiten ver rápidamente cuánto cambia F1, recall, precisión, ROC-AUC o AP para cada categoría.',
            },
            {
                term: '3. Usa el delta como señal, no aislado',
                definition:
                    'Un delta positivo indica degradación porque se calcula como original menos tras el ataque. Contrástalo siempre con el soporte y con el valor absoluto de la métrica.',
            },
        ],
        note: 'Una métrica puede aparecer como no definida cuando faltan predicciones positivas, positivos reales o negativos suficientes para calcularla con sentido.',
    },
    matrices: {
        title: 'Cómo leer las matrices de confusión',
        summary:
            'Sirven para localizar qué confusiones comete el modelo y comprobar si el ataque desplaza predicciones correctas hacia clases equivocadas.',
        items: [
            {
                term: '1. Localiza la diagonal',
                definition:
                    'Cada fila es la clase real y cada columna la predicha. La diagonal contiene aciertos; cuanto más valor se concentre fuera de ella, más confusiones existen.',
            },
            {
                term: '2. Compara original y adversaria',
                definition:
                    'La matriz original actúa como referencia. Si tras el ataque aparecen nuevos valores fuera de la diagonal puedes identificar exactamente hacia qué clases se desplazan los errores.',
            },
            {
                term: '3. Elige conteos o porcentajes',
                definition:
                    'Los conteos muestran cuántas imágenes representa cada celda. La versión normalizada facilita comparar clases con tamaños distintos al expresar cada fila respecto a su clase real.',
            },
        ],
        note: 'La vista compacta oculta filas sin muestras reales para facilitar la exploración; la vista completa conserva todo el catálogo de clases.',
    },
    curves: {
        title: 'Cómo leer las curvas de clasificación',
        summary:
            'Las curvas muestran cómo cambia el comportamiento al variar el umbral de decisión. Compara la serie original con la adversaria y observa si la curva y su puntuación se deterioran.',
        items: [
            {
                term: '1. Elige el ámbito',
                definition:
                    'Global (micro) combina las decisiones de todas las clases y ofrece una visión agregada. Elegir una clase permite estudiar esa categoría frente al resto.',
            },
            {
                term: '2. Interpreta la curva adecuada',
                definition:
                    'ROC relaciona verdaderos y falsos positivos. Precision–Recall muestra el equilibrio entre recuperar positivos y mantener precisión, y suele ser especialmente informativa en clases poco frecuentes.',
            },
            {
                term: '3. Compara las puntuaciones',
                definition:
                    'ROC-AUC y AP resumen cada curva en un valor. Cuanto mayor sea, mejor. La diferencia original − tras el ataque cuantifica la degradación de esa capacidad.',
            },
        ],
        note: 'Las curvas pueden no estar definidas si una clase no contiene positivos o negativos suficientes. El promedio macro da el mismo peso a todas las clases disponibles.',
    },
    images: {
        title: 'Cómo leer los resultados por imagen',
        summary:
            'Utiliza los casos individuales para conectar las métricas globales con ejemplos concretos: qué predicción cambia, cuánto cae la confianza y dónde se fija el modelo.',
        items: [
            {
                term: '1. Empieza por la transición',
                definition:
                    'Correcta → incorrecta representa un éxito adversarial; incorrecta → correcta es una recuperación. Los filtros permiten localizar rápidamente estos casos.',
            },
            {
                term: '2. Compara imagen y perturbación',
                definition:
                    'La vista original, la adversaria y el mapa de perturbación ayudan a comprobar si el cambio visual es perceptible y a relacionarlo con L∞ y L2.',
            },
            {
                term: '3. Usa Grad-CAM como apoyo interpretativo',
                definition:
                    'Grad-CAM resalta regiones influyentes antes y después del ataque. Puede ayudar a explicar cambios de atención, pero no demuestra causalidad por sí solo.',
            },
        ],
        note: 'Los filtros se aplican sobre todas las imágenes evaluadas; la paginación solo limita cuántos resultados se muestran simultáneamente.',
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
