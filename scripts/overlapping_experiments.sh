#/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
python usage.py --strings\
            "\"5 \"utilizzando una modulistica conforme a quella pubblicata sul sito istituzionale del Ministero dell'interno\" DL 5/2012 - Semplificazione e sviluppo Art\""\
            --cuda