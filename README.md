# Team_15

Ce projet est développé dans le cadre de l’événement **Défi des 100 jours de Ran.données 2021** organisé par l'AÉBINUM et IVADO.

### Introduction :
Le cancer du cerveau est parmi les types de cancer les plus critiques. Plusieurs symptômes peuvent apparaître en fonction de la localisation du cancer dans le cerveau. Le gliome représente la principale et la plus fréquente tumeur cérébrale primaire maligne (*Ostrom QT et al.*). Son pronostic est mauvais avec une survie médiane variables allant de 1 an à 7 ans selon le grade et le sous-type de la tumeur (société canadienne du cancer).

### Données :
Les données sont de l'imagerie IRM des gliomes de bas grade obtenues à partir de 110 patients de 5 institutions et stockées dans la base de données *TCGA*. Elles ont été décrites dans :
*Buda, Mateusz, Ashirbani Saha, and Maciej A. Mazurowski. "Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm." Computers in biology and medicine 109 (2019): 218-225.*

### Objectif :
Le but de ce travail est d'utiliser les réseaux de neurones convolutifs (CNN) un sous type d'apprentissage profond (*Deep Learning*) pour distinguer les patients ayant des tumeurs de ceux sains en utilisant les images IRM. Puis de faire de la segmentation afin d'identifier dans quelle région de ces images se localise la tumeur. Enfin intégrer ces données avec des données cliniques pour améliorer la fiabilité du workflow de diagnostic et aider le personnel médical à faire de meilleures prédictions.
