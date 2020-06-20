# VOPDI-UNLP
Visual Odometry. Seguimiento de puntos clave en imágenes, cálculo de movimiento y actitud de cámara

##Trabajo realizado por MITIDIERI Pedro y CONCIA Bernardo (DGNC - Departamento de Aeronáutica) para el curso de posgrado de Procesamiento Digital de Imágenes de la UNLP

Se encuentra la explicación del algoritmo y la ayuda de algunos comando de OpenCV en el archivo VO con Opencv⁄C++.pdf

## Código
Se encuentran tres carpetas con el codigo para la implementacion del algoritmo. Para su implementación se debe contar con OpenCv 3.0 o posterior.

__Actitud:__  El programa imprime en pantalla el ángulo actual con respecto al angulo incial (0,0,0). (ángulos de Euler). El codigo esta desordenado ya que solo se hizo una versión para comprobar el funcionamiento.

__TrayectoriaPlanoLive:__ El programa calcula el desplazamiento de los puntos claves y lo grafica en pantalla adquiriendo datos de la camara web.

__TrayectoriaPlanoVideo:__ El programa calcula el desplazamiento de los puntos claves y lo grafica en pantalla desde un video que puede seleccionarse desde la ejecución del programa.

