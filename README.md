# ASR-and-classification

### Ceci est un projet permettant de faire la classification d'un audio

###### Le processus est tout d'abord la transcription des audios en textes

###### Pour ce faire on a directement utilisé le model disponible sur huggingface qui s'appelle jonatasgrosman/wav2vec2-large-xlsr-53-french

###### En suite on fait le fine-tuning du model camembert-base

##### Pour éxecuter et tester le pipeline complet, il faut installer les dépendances se trouvant au niveau du fichier requirements.txt en utilisant la commande << pip install -r requirements.txt >>

##### Au niveau de la ligne 18 du fichier app.py il faudra remplacer le "path" par le chemin menant vers le fichier audio et executer la commande python app.py