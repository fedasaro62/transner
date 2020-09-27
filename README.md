# transner
NER with transformer

* input: list of strings
* output: dict object containing the extracted entities

* example of usage:
```console
$python usage.py --strings "Mario Rossi è nato a Busto Arsizio" \
                     "Il signor Di Marzio ha effettuato un pagamento a Matteo" \
                     "Marco e Luca sono andati a Magenta"

{
  "results": [
    {
      "entities": [
        {
          "offset": 0,
          "type": "PERSON",
          "value": "mario rossi"
        },
        {
          "offset": 21,
          "type": "LOCATION",
          "value": "busto arsizio"
        }
      ],
      "sentence": "Mario Rossi è nato a Busto Arsizio"
    },
    {
      "entities": [
        {
          "offset": 0,
          "type": "PERSON",
          "value": "il signor d'alberto"
        },
        {
          "offset": 49,
          "type": "PERSON",
          "value": "matteo"
        }
      ],
      "sentence": "Il signor D'Alberto ha effettuato un pagamento a Matteo"
    },
    {
      "entities": [
        {
          "offset": 0,
          "type": "PERSON",
          "value": "marco"
        },
        {
          "offset": 8,
          "type": "PERSON",
          "value": "luca"
        },
        {
          "offset": 27,
          "type": "LOCATION",
          "value": "magenta"
        }
      ],
      "sentence": "Marco e Luca sono andati a Magenta"
    }
  ],
  "timestamp": 1581065432.7972977
}
```

## HOW TO USE:
Download one of the pretrained model and move it to `transner/` folder..<br> or use the one in the repo after installing LFS:
`git lfs install`
pretrained models link: https://istitutoboella-my.sharepoint.com/:f:/g/personal/matteo_senese_linksfoundation_com/EvhOF23tja5Nuo3mw03v24oB7D14q9cjk16Ca7xF3nTm-A?e=AWpuiu
