Qué hace el script                                                                                                                                                                                        
                                                                                                                                                                                                            
  Imagina que tienes un niño que no sabe inglés y quieres enseñarle a traducir del español al inglés.                                                                                                       
                  
  Lo que hace el script es exactamente eso, pero con una red neuronal:                                                                                                                                      
                  
  ---
  1. Prepara el "cerebro" (la arquitectura Transformer)

  Crea una red neuronal con millones de parámetros (números ajustables). Al principio todos son aleatorios, el modelo no sabe nada. Es como un niño recién nacido.

  ---
  2. Le da un diccionario (los vocabularios)

  Le dice: "estas son las 5000 palabras en español que vas a ver, y estas son las 5000 en inglés que puedes escribir". Cada palabra tiene un número asignado.

  ---
  3. Lo entrena (el bucle de épocas)

  Le muestra 130.000 pares de frases reales de libros:
  - "El gato duerme" → "The cat sleeps"
  - "Buenos días" → "Good morning"
  - ... 130.000 veces más

  En cada par hace esto:
  1. Le da la frase en español
  2. Le dice "intenta traducirla"
  3. Compara lo que dijo con la traducción correcta
  4. Calcula el error (loss)
  5. Ajusta los millones de parámetros un poquito para que la próxima vez falle menos
  6. Repite

  Después de ver todos los pares miles de veces, los parámetros se van ajustando solos hasta que el modelo aprende los patrones del idioma.

  ---
  El loss bajando = está aprendiendo

  Época 1:  Loss 8.5  → "no tengo ni idea, adivino aleatoriamente"
  Época 5:  Loss 5.2  → "algo he captado"
  Época 10: Loss 3.1  → "entiendo bastantes patrones"

  ---
  Qué podemos hacer con él

  Una vez entrenado, hay tres cosas interesantes:

  1. Traducir frases nuevas
  Darle una frase en español que nunca ha visto y que genere la traducción en inglés token por token.

  2. Ver qué ha aprendido (visualizar la atención)
  El Transformer tiene matrices de atención que puedes visualizar como un mapa de calor. Te muestra qué palabra española estaba "mirando" el modelo cuando generó cada palabra inglesa. Es fascinante ver
  cómo conecta "gato" con "cat".

  3. Guardarlo y cargarlo
  Guardar los pesos entrenados en un archivo para no tener que reentrenar cada vez.

  ---
  Lo que NO puede hacer (limitaciones de este modelo)

  - Con solo 5000 palabras en el vocabulario, muchas palabras aparecerán como <UNK> (desconocida)
  - Con 10 épocas de entrenamiento, las traducciones serán mediocres pero reconocibles
  - No tiene mecanismo de búsqueda (beam search), solo escoge la palabra más probable en cada paso
