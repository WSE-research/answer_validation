## File description

`qald_labels.json` -- dataset with labels for each answer (and for each question). Structure:

```json
{
    "0": [ // key is a number of question. the order is the same as in train + test
        [ // list of entities retrieved for each answer candidates
            "Magazines disestablished in 2004",
            "Defunct magazines published in the United States",
            "Wargaming magazines",
            "Magazines established in 1996"
        ],
```

`qald-qa-dataset.json` -- dataset with Question and Answer (labels) representation. Structure:

```json
{
  "0": [ // key is a number of question. the order is the same as in train + test
    [ // list of answer candidates
      0, // 0 -- if wrong answer, 1 -- if right answer
      [
        "Textual Question",
        "Answer text generated with labels"
      ]
    ],
  ],
}
```
