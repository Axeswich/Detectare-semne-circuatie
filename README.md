# Detectare-semne-circuatie

1.Fisierul Image_Recognition_Tester.py testeaza modelul creat de Trainer si creeaza conexiunea intre model si site-ul hostat local.
2.Fisierul Image_Recognition_Trainer.py antreneaza modelul folosind dataset-ul German Traffic Sign Recognition Benchmark(GTSRB), folosind peste 50000 de imagini pentru a face asta.
3.Fisierul index.html este folosit pentru creearea site-ului web, care afiseaza un panel unde se poate incarca o poza(poate sa aiba orice format), iar site-ul, folosind fisierle de mai sus, va verifica, iar dupa va afisa in scris ce semn este.

In viitor, dorim sa implementam aplicatia noastra, ca sa o putem integra in industria automotiva.Scopul aplicatiei noastre este sa minimizam accidentele si sa facem condusul cat mai usor pentru toata lumea.

Librarii folosite:Tensorflow, Flask, Keras, Sklearn, Numpy, Pillow.
Link catre Dataset:https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
