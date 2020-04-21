# transner
NER with transformer


## route: /transner/v0.3/ner
* input: JSON object containing a list of strings {“strings”: [...]}
    * This interface expects sentences taken eventually from longer documents or records from a table. Please check with Pipple if they are willing to contribute to the provision of a sentence splitter for longer documents. Otherwise, we will implement it ourselves
* output: JSON object containing the extracted entities

* example of usage:
```console
$ curl -i -H "Content-Type: application/json" -X POST -d '{"strings": ["Mario Rossi è nato a Busto Arsizio", "Il signor Di Marzio ha effettuato un pagamento a Matteo", "Marco e Luca sono andati a Magenta"]}' http://localhost:5000/ner_api/v0.1/ner

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
Download one of the pretrained model and move it to `rest/` folder. Select appropriately its path in the `rest/config.py` in `PRETRAINED_MODEL`.<br>
pretrained models link: https://istitutoboella-my.sharepoint.com/:f:/g/personal/matteo_senese_linksfoundation_com/EvhOF23tja5Nuo3mw03v24oB7D14q9cjk16Ca7xF3nTm-A?e=AWpuiu
