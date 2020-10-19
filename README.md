# be-PDFDecoder
Aplicacion desarrollada en la Hackathon BBVA 2020. Deteccion de variables financieras a partir de un documento pdf.

## Pasos para correr la aplicación localmente:

1. Build the container locally: `docker build -t my-flask-app .`
2. Run the container locally: `docker run -it --rm -p 80:80 --name my-flask-app -v $(pwd)/app:/app my-flask-app`
3. [http://localhost](http://localhost)

## Aplicacion
Puedes ver el repositorio del frontend aqui https://github.com/Leinadh/fe-PDFDecoder

Esta aplicacion consiste en detectar variables financieras como son activos, pasivos, patrimonios, etc de PYMES que desean solicitar algun servicio del banco. Actualmente la deteccion de estas se realizar de manera manual. Por ello, es necesario automatizar el proceso mencionado. Para ello, se uso el servicio de AWS Textract como OCR(Optical Character Recognition). Luego, a partir de algoritmos de machine learning y analisis de texto, se detectaron las variables correspondientes.  

En promedio, se logro detectar 8 variables de las 12. Ademas, el tiempo de respuesta del modelo fue de 15 segundos comparado al tiempo promedio de un asesor que se encarga de esta tarea que es de 7 minutos
