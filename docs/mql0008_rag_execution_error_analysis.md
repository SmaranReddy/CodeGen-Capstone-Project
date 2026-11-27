# MQL-0008: forensic analysis of the hybrid RAG execution errors

**Status: ANALYSIS COMPLETE — all 29 cases explained with evidence.**

Read-only investigation, 2026-11-17. Source: `outputs/nl2mql_mql0008_approved_corpus_v2_1/scoring_runs/run_20261117T053551Z/` (per-record `evaluation_results.jsonl` of both arms, the scored `predictions/` with retrieval metadata) and the approved corpus v2.1 shipped in `kaggle/nl2mql_mql0008_approved_corpus_v2_1/`. Atlas was not contacted, nothing was re-scored or regenerated, and no existing file was changed. This file is the only output.

## 1. Executive summary

- **None of the 29 errors is an Atlas limit, a timeout, a result-size cap or a type mismatch in the data.** All 29 are pipelines that MongoDB rejects because of how the model wrote them: 23 are rejected when the pipeline is parsed, 6 fail when the first document is processed.
- **Zero-shot was already wrong on most of them.** On these 29 items zero-shot was correct on 6, incorrect on 21 and also failed on 2. So the rise from 13 to 29 mostly converts *silent wrong answers* into *loud failures*; only 6 of the 29 are real accuracy losses.
- **In the other direction**, of zero-shot's 13 execution errors the RAG arm fixed 11 (6 became correct, 5 executed but were incorrect) and 2 failed in both arms.
- **The largest single cause (8 of 29)** is a comparison written in aggregation-expression form inside `$match` without `$expr` (`unknown top level operator`). This pattern occurs 8 times in the RAG arm, always fatally, and **never** in zero-shot or in the corpus.
- **Retrieval changes how the model writes MQL, and about two thirds of the errors (19 of 29) sit in that changed style.** 203 of 892 RAG outputs use the corpus join alias `__jN` (zero-shot: 0), 69 use `$expr` (zero-shot: 6), 323 use `$unwind` (zero-shot: 172). The model imitates the corpus's more advanced constructs and sometimes imitates them incompletely. The corpus examples themselves are not faulty: no corpus record contains any of the failing constructs.
- **Retrieval signal per case:** strong in 6, plausible in 13, none identifiable in 10. In no case did a retrieved example contain the error itself.
- **The converter and the corpus are not implicated.** No converter or corpus change is recommended.

## 2. Root-cause distribution

Each case has exactly one category; the counts sum to 29.

| Category | Cases | Evaluation ids |
|---|---:|---|
| C1. Aggregation-expression operator used as a query operator in `$match` without `$expr` | 8 | 0171, 0380, 0559, 0898, 0899, 0900, 0901, 0918 |
| C3. Pipeline stage name written without `$` | 4 | 0720, 0737, 0738, 0739 |
| C2. `$group` output field that is not an accumulator | 3 | 0090, 0093, 0932 |
| C7. `$size` applied to a field that does not exist | 3 | 0175, 0922, 0923 |
| C5. `$unwind` path without the `$` prefix | 2 | 0044, 1021 |
| C6. `$lookup` with `let`/`pipeline`: `$in` given a scalar instead of an array | 2 | 0042, 0755 |
| C8. Expression operator used as a `$group` accumulator | 2 | 0258, 0839 |
| C9. Pipeline stages nested inside a query `$in` list (subquery emulation) | 2 | 0364, 0365 |
| C4. Pipeline element is a data document, not a stage | 1 | 0514 |
| C10. Reference to an undefined `$$` variable | 1 | 0168 |
| C11. `$filter` applied to a document (`$$ROOT`) instead of an array | 1 | 0974 |
| **Total** | **29** | |

Grouped (same cases, coarser view):

| Group | Cases |
|---|---:|
| G1 query/expression language confusion | 8 |
| G2 `$group` accumulator misuse | 5 |
| G3 malformed stage or path syntax | 7 |
| G4 reference to something that does not exist | 5 |
| G5 subquery emulation | 4 |
| **Total** | **29** |

### Counts requested

| Quantity | Count |
|---|---:|
| RAG execution-error cases accounted for | 29 of 29 |
| Also an execution error in zero-shot | 2 (spider_dev_0044, spider_dev_0364) |
| Introduced in RAG where zero-shot executed | 27 (6 zero-shot correct, 21 zero-shot incorrect) |
| Same underlying issue in both arms | 3 (spider_dev_0364 nested stages under `$in`; spider_dev_0922, spider_dev_0923 invented field `treatments`) |
| Strong retrieval-related signal | 6 |
| Plausible retrieval association only | 13 |
| No identifiable retrieval relationship | 10 |
| Insufficient evidence to determine the cause | 0 |
| Rejected at parse time (detectable without data) | 23 |
| Fail only when documents are processed | 6 (C6, C7, C11) |

The "same underlying issue" row overlaps with the rows above it: spider_dev_0364 is also one of the two that failed in both arms, and spider_dev_0922/0923 are among the 27 where zero-shot executed. The three retrieval rows are mutually exclusive and sum to 29.

By hardness: hard 10, medium 10, extra 7, easy 2. By database: car_1 5, world_1 5, dog_kennels 5, network_1 4, cre_Doc_Template_Mgt 3, concert_singer 2, student_transcripts_tracking 2, flight_2 1, orchestra 1, singer 1.

## 3. Paired zero-shot versus RAG diagnostics

Full MQL of both arms for every row is in section 4.

| Evaluation id | Zero-shot outcome | RAG outcome | Category | Retrieved example ids | Retrieval signal | Corrective action |
|---|---|---|---|---|---|---|
| spider_dev_0042 | incorrect (wrong_row_count) | execution error | C6 | 02511, 00203, 06549 | none | F1 |
| spider_dev_0044 | execution error | execution error | C5 | 06794, 02154, 04092 | strong | F1, F2 |
| spider_dev_0090 | incorrect (wrong_values) | execution error | C2 | 04087, 00261, 01538 | plausible | F1, F2 |
| spider_dev_0093 | correct | execution error | C2 | 00500, 02673, 00585 | plausible | F1, F2 |
| spider_dev_0168 | incorrect (wrong_row_count) | execution error | C10 | 04516, 05504, 00540 | none | F1 |
| spider_dev_0171 | incorrect (wrong_row_count) | execution error | C1 | 02362, 03729, 04582 | strong | F1, F2, F3 |
| spider_dev_0175 | incorrect (wrong_row_count) | execution error | C7 | 06588, 02942, 06580 | strong | F1 |
| spider_dev_0258 | incorrect (wrong_column_count) | execution error | C8 | 02276, 01046, 02787 | none | F1, F2 |
| spider_dev_0364 | execution error | execution error | C9 | 00253, 02313, 02370 | plausible | F1, F2 |
| spider_dev_0365 | incorrect (wrong_row_count) | execution error | C9 | 06753, 00413, 01605 | plausible | F1, F2 |
| spider_dev_0380 | correct | execution error | C1 | 05543, 04530, 06337 | plausible | F1, F2 |
| spider_dev_0514 | correct | execution error | C4 | 02337, 02339, 04224 | none | F1 |
| spider_dev_0559 | correct | execution error | C1 | 03803, 01063, 02952 | plausible | F1, F2 |
| spider_dev_0720 | correct | execution error | C3 | 00119, 02277, 06819 | none | F1, F2 |
| spider_dev_0737 | incorrect (wrong_row_count) | execution error | C3 | 00130, 00595, 01325 | none | F1, F2 |
| spider_dev_0738 | incorrect (wrong_row_count) | execution error | C3 | 03688, 01334, 00277 | none | F1, F2 |
| spider_dev_0739 | incorrect (wrong_row_count) | execution error | C3 | 00130, 00595, 01325 | none | F1, F2 |
| spider_dev_0755 | incorrect (wrong_row_count) | execution error | C6 | 02284, 03287, 04580 | none | F1 |
| spider_dev_0839 | incorrect (wrong_column_count) | execution error | C8 | 02448, 00382, 03999 | none | F1, F2 |
| spider_dev_0898 | incorrect (wrong_values) | execution error | C1 | 04999, 04931, 05991 | plausible | F1, F2 |
| spider_dev_0899 | incorrect (wrong_row_count) | execution error | C1 | 00688, 03589, 05485 | plausible | F1, F2 |
| spider_dev_0900 | incorrect (wrong_values) | execution error | C1 | 01468, 03309, 02132 | plausible | F1, F2 |
| spider_dev_0901 | incorrect (wrong_values) | execution error | C1 | 00688, 03589, 05485 | plausible | F1, F2 |
| spider_dev_0918 | incorrect (wrong_row_count) | execution error | C1 | 06656, 02773, 00334 | strong | F1, F2, F3 |
| spider_dev_0922 | incorrect (wrong_row_count) | execution error | C7 | 02954, 02787, 03132 | plausible | F1 |
| spider_dev_0923 | incorrect (wrong_row_count) | execution error | C7 | 02292, 03062, 02787 | plausible | F1 |
| spider_dev_0932 | incorrect (wrong_row_count) | execution error | C2 | 02930, 02942, 02786 | plausible | F1, F2 |
| spider_dev_0974 | incorrect (wrong_row_count) | execution error | C11 | 02946, 00689, 05672 | strong | F1 |
| spider_dev_1021 | correct | execution error | C5 | 04015, 00628, 02079 | strong | F1, F2 |

Corrective actions: F1 = generation-time validate-and-retry (recommendation 1); F2 = prompt rule block (recommendation 2); F3 = example display form (recommendation 3).

### Patterns shared with the 13 zero-shot execution errors

| Pattern | Zero-shot cases | RAG cases | Same items? |
|---|---|---|---|
| Stage name without `$` (Atlas code 8000) | 3 (0197, 0511, 0820) | 4 (0720, 0737, 0738, 0739) | no; RAG fixed the three zero-shot items (1 correct, 2 incorrect) and produced four new ones |
| Stages nested under a query `$in` | 4 (0363, 0364, 0366, 0496) | 2 (0364, 0365) | 0364 in both; RAG made 0363, 0366 and 0496 correct |
| Expression operator as `$group` accumulator (`$subtract`) | 1 (0837) | 1 (0839) | no |
| Invented array field (`treatments`) | executes, scored incorrect (0922, 0923) | execution error (0922, 0923) | yes |
| `$size` with a query operator (`$size: {$gte: 2}`) | 2 (0525, 0943) | 0 | zero-shot only |
| Inclusion and exclusion mixed in `$project` | 2 (0044, 1027) | 0 | zero-shot only |
| Bare expression operator in `$match` (C1) | 0 | 8 | RAG only |
| `$unwind` path without `$` (C5) | 0 | 2 | RAG only |

The remaining zero-shot error (spider_dev_0941, `$multiply` on an array) has no RAG counterpart.

## 4. Complete case analysis (29 records)

For each case: the question, the saved Atlas message (the scorer keeps the exception type and the first 300 characters of the message), the generated MQL of both arms as parsed by the scorer, the three retrieved examples with rank and score, and the finding.

### 1. spider_dev_0042 (concert_singer, extra)

- **Question:** What are the names and locations of the stadiums that had concerts that occurred in both 2014 and 2015?
- **Atlas error (code beyond the saved 300 characters):** `PlanExecutor error during aggregation :: caused by :: error while multiplanner was selecting best plan :: caused by :: $in requires an array as a second argument, found: int`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"$in":["$Stadium_ID","$$stadium_id"]}` inside the `$lookup` sub-pipeline
- **Root cause:** C6. `$lookup` with `let`/`pipeline`: `$in` given a scalar instead of an array. Confidence: high.
- **Explanation:** `$in` needs an array as its second argument; `$$stadium_id` is a number. `$eq` was meant. The retrieved examples all use the `localField`/`foreignField` form; the approved corpus contains no `let`/`pipeline` lookup at all, so this construct did not come from an example.
- **Origin:** generation. **Retrieval signal:** none. **Corrective action:** F1.
- **Zero-shot:** incorrect (wrong_row_count) — row count 0 != gold 1

RAG MQL:

```json
{"collection":"concert_singer__stadium","pipeline":[{"$lookup":{"from":"concert_singer__concert","let":{"stadium_id":"$Stadium_ID"},"pipeline":[{"$match":{"$expr":{"$in":["$Stadium_ID","$$stadium_id"]}}},{"$match":{"Year":{"$in":["2014","2015"]}}}],"as":"concerts"}},{"$unwind":"$concerts"},{"$group":{"_id":"$Name","locations":{"$addToSet":"$Location"}}},{"$match":{"locations":{"$size":2}}},{"$project":{"_id":0,"Name":"$_id","Location":{"$first":"$locations"}}}]}
```

Zero-shot MQL:

```json
{"collection":"concert_singer__stadium","pipeline":[{"$lookup":{"from":"concert_singer__concert","localField":"Stadium_ID","foreignField":"Stadium_ID","as":"concerts"}},{"$match":{"concerts.Year":{"$in":["2014","2015"]}}},{"$group":{"_id":"$Name","locations":{"$addToSet":"$Location"}}},{"$project":{"_id":0,"name":"$_id","locations":1}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.323 | spider_train_02511 | movie_1 | What are the titles of all movies that have between 3 and 5 stars? |
| 2 | 1.213 | spider_train_00203 | bike_1 | What are the id and name of the stations that have ever had more than 12 bikes available? |
| 3 | 1.189 | spider_train_06549 | wine_1 | What are the wines that have prices higher than 50 and made of Red color grapes? |

### 2. spider_dev_0044 (concert_singer, hard)

- **Question:** What are the number of concerts that occurred in the stadium with the largest capacity ?
- **Atlas error (code 28818):** `path option to $unwind stage should be prefixed with a '$': __j1`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"$unwind":"__j1"}`
- **Root cause:** C5. `$unwind` path without the `$` prefix. Confidence: high.
- **Explanation:** The path lacks its `$`. The alias `__j1` exists only in corpus examples (two of the three retrieved examples contain `{"$unwind":"$__j1"}`); zero-shot never emits it. The join was copied from an example and the `$` was dropped while copying.
- **Origin:** generation (miscopy of a retrieved pattern). **Retrieval signal:** strong. **Corrective action:** F1, F2.
- **Zero-shot:** execution error — `Invalid $project :: caused by :: Cannot do inclusion on field Capacity in exclusion projection`

RAG MQL:

```json
{"collection":"concert_singer__stadium","pipeline":[{"$sort":{"Capacity":-1}},{"$limit":1},{"$lookup":{"from":"concert_singer__concert","localField":"Stadium_ID","foreignField":"Stadium_ID","as":"__j1"}},{"$unwind":"__j1"},{"$group":{"_id":null,"count":{"$sum":1}}},{"$project":{"_id":0,"count":1}}]}
```

Zero-shot MQL:

```json
{"collection":"concert_singer__stadium","pipeline":[{"$sort":{"Capacity":-1}},{"$limit":1},{"$lookup":{"from":"concert_singer__concert","localField":"Stadium_ID","foreignField":"Stadium_ID","as":"concerts"}},{"$project":{"concerts":0,"Capacity":1}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.626 | spider_train_06794 | activity_1 | Find the name of the activity that has the largest number of student participants. |
| 2 | 1.500 | spider_train_02154 | formula_1 | What is the name of the race that occurred most recently? |
| 3 | 1.486 | spider_train_04092 | student_1 | Find the name of the teacher who teaches the largest number of students. |

### 3. spider_dev_0090 (car_1, medium)

- **Question:** For each continent, list its id, name, and how many countries it has?
- **Atlas error (code 40234):** `The field 'Continent' must be an accumulator object`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `"Continent":"$Continent"` inside `$group`
- **Root cause:** C2. `$group` output field that is not an accumulator. Confidence: high.
- **Explanation:** Every non-`_id` field of `$group` must be an accumulator; `{"$first":"$Continent"}` was needed (zero-shot wrote exactly that). The retrieved examples are single-table counts whose `$project` renames fields as `"alias":"$_id.g0"`; the output uses that rename style in the wrong stage. Association only.
- **Origin:** generation. **Retrieval signal:** plausible. **Corrective action:** F1, F2.
- **Zero-shot:** incorrect (wrong_values) — some predicted column matches no gold column

RAG MQL:

```json
{"collection":"car_1__continents","pipeline":[{"$lookup":{"from":"car_1__countries","localField":"ContId","foreignField":"Continent","as":"country"}},{"$unwind":"$country"},{"$group":{"_id":"$ContId","Continent":"$Continent","countryCount":{"$sum":1}}},{"$project":{"_id":0,"ContId":"$_id","Continent":1,"countryCount":1}}]}
```

Zero-shot MQL:

```json
{"collection":"car_1__continents","pipeline":[{"$lookup":{"from":"car_1__countries","localField":"ContId","foreignField":"Continent","as":"countries"}},{"$group":{"_id":"$ContId","Continent":{"$first":"$Continent"},"CountryCount":{"$sum":1}}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.667 | spider_train_04087 | student_1 | For each classroom, show the classroom number and find how many students are using it. |
| 2 | 1.638 | spider_train_00261 | musical | How many musicals has each nominee been nominated for? |
| 3 | 1.553 | spider_train_01538 | insurance_and_eClaims | For each policy type, return its type code and its count in the record. |

### 4. spider_dev_0093 (car_1, medium)

- **Question:** How many models does each car maker produce? List maker full name, id and the number.
- **Atlas error (code 40234):** `The field 'MakerId' must be an accumulator object`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `"MakerId":"$Id"` inside `$group`
- **Root cause:** C2. `$group` output field that is not an accumulator. Confidence: high.
- **Explanation:** Same defect as spider_dev_0090. Zero-shot used `{"$first":"$Id"}` and was correct, so this is a real loss.
- **Origin:** generation. **Retrieval signal:** plausible. **Corrective action:** F1, F2.
- **Zero-shot:** correct — multiset match

RAG MQL:

```json
{"collection":"car_1__car_makers","pipeline":[{"$lookup":{"from":"car_1__model_list","localField":"Id","foreignField":"Maker","as":"models"}},{"$unwind":"$models"},{"$group":{"_id":"$FullName","MakerId":"$Id","ModelCount":{"$sum":1}}},{"$project":{"_id":0,"MakerFullname":"$_id","MakerId":"$MakerId","ModelCount":"$ModelCount"}}]}
```

Zero-shot MQL:

```json
{"collection":"car_1__car_makers","pipeline":[{"$lookup":{"from":"car_1__model_list","localField":"Id","foreignField":"Maker","as":"models"}},{"$unwind":"$models"},{"$group":{"_id":"$FullName","Id":{"$first":"$Id"},"ModelCount":{"$sum":1}}},{"$sort":{"ModelCount":-1}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.667 | spider_train_00500 | allergy_1 | How many students does each advisor have? |
| 2 | 1.557 | spider_train_02673 | party_host | How many hosts does each nationality have? List the nationality and the count. |
| 3 | 1.438 | spider_train_00585 | store_1 | How many customers does Steve Johnson support? |

### 5. spider_dev_0168 (car_1, extra)

- **Question:** Among the cars that do not have the minimum horsepower , what are the make ids and names of all those with less than 4 cylinders ?
- **Atlas error (code 17276):** `Use of undefined variable: minHorsepower`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `"$$minHorsepower"`
- **Root cause:** C10. Reference to an undefined `$$` variable. Confidence: high.
- **Explanation:** The variable is never defined (no `let`, no earlier stage computes the minimum). The minimum-horsepower subquery was not built. The corpus contains no user `$$` variable.
- **Origin:** generation. **Retrieval signal:** none. **Corrective action:** F1.
- **Zero-shot:** incorrect (wrong_row_count) — row count 8 != gold 3

RAG MQL:

```json
{"collection":"car_1__cars_data","pipeline":[{"$lookup":{"from":"car_1__car_names","localField":"MakeId","foreignField":"Id","as":"car_name"}},{"$unwind":"$car_name"},{"$lookup":{"from":"car_1__car_makers","localField":"car_name.Make","foreignField":"Maker","as":"car_maker"}},{"$unwind":"$car_maker"},{"$match":{"$expr":{"$ne":["$Horsepower","$$minHorsepower"]}}},{"$group":{"_id":"$car_maker.Id","make_name":{"$first":"$car_maker.FullName"},"count":{"$sum":1}}},{"$match":{"count":{"$lt":4}}},{"$project":{"_id":0,"make_id":"$_id","make_name":1}}]}
```

Zero-shot MQL:

```json
{"collection":"car_1__cars_data","pipeline":[{"$match":{"Cylinders":{"$lt":4}}},{"$lookup":{"from":"car_1__car_names","localField":"MakeId","foreignField":"Id","as":"car_name"}},{"$unwind":"$car_name"},{"$lookup":{"from":"car_1__car_makers","localField":"car_name.Make","foreignField":"Maker","as":"car_maker"}},{"$unwind":"$car_maker"},{"$project":{"_id":0,"MakeId":"$car_name.MakeId","MakeName":"$car_maker.Maker"}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.349 | spider_train_04516 | document_management | Return the codes of the document types that do not have a total access count of over 10000. |
| 2 | 1.333 | spider_train_05504 | voter_2 | Report all majors that have less than 3 students. |
| 3 | 1.182 | spider_train_00540 | store_1 | What are the names of the top 8 countries by total invoice size and what are those sizes? |

### 6. spider_dev_0171 (car_1, extra)

- **Question:** Which models are lighter than 3500 but not built by the 'Ford Motor Company'?
- **Atlas error (code beyond the saved 300 characters):** `unknown top level operator: $lt. If you have a field name that starts with a '$' symbol, consider using $getField or $setField.`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"$match":{"$and":[{"$lt":["$__j1.Weight",3500]},{"$ne":["$Make","Ford Motor Company"]}]}}`
- **Root cause:** C1. Aggregation-expression operator used as a query operator in `$match` without `$expr`. Confidence: high.
- **Explanation:** `{"$lt":[a,b]}` is the aggregation-expression form and is only legal under `$expr`. Retrieved example spider_train_04582 shows exactly this array form under `{"$match":{"$expr":{"$and":[...]}}}`; the output reproduces the form and omits `$expr`. It also uses the corpus alias `__j1`.
- **Origin:** generation (incomplete imitation of a retrieved pattern). **Retrieval signal:** strong. **Corrective action:** F1, F2, F3.
- **Zero-shot:** incorrect (wrong_row_count) — row count 406 != gold 26

RAG MQL:

```json
{"collection":"car_1__car_names","pipeline":[{"$lookup":{"from":"car_1__cars_data","localField":"MakeId","foreignField":"Id","as":"__j1"}},{"$unwind":"$__j1"},{"$match":{"$and":[{"$lt":["$__j1.Weight",3500]},{"$ne":["$Make","Ford Motor Company"]}]}},{"$project":{"_id":0,"Model":1}}]}
```

Zero-shot MQL:

```json
{"collection":"car_1__car_names","pipeline":[{"$lookup":{"from":"car_1__model_list","localField":"Make","foreignField":"Maker","as":"model_list"}},{"$lookup":{"from":"car_1__cars_data","localField":"Id","foreignField":"MakeId","as":"cars_data"}},{"$match":{"model_list.Maker":{"$ne":"ford"},"cars_data.Weight":{"$lt":3500}}},{"$project":{"Model":1,"_id":0}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.000 | spider_train_02362 | csu_1 | What campus had more than 400 total enrollment but more than 200 full time enrollment in year 1956? |
| 2 | 0.968 | spider_train_03729 | program_share | Which channels are not owned by CCTV? Give me the channel names. |
| 3 | 0.947 | spider_train_04582 | company_office | Which headquarter locations are used by more than 2 companies? |

### 7. spider_dev_0175 (car_1, extra)

- **Question:** Which are the car makers which produce at least 2 models and more than 3 car makers ? List the id and the maker .
- **Atlas error (code 17124):** `PlanExecutor error during aggregation :: caused by :: The argument to $size must be an array, but was of type: missing`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"$size":"$__j1"}` after `$unwind` and `$group`
- **Root cause:** C7. `$size` applied to a field that does not exist. Confidence: high.
- **Explanation:** After `$group` the field `__j1` no longer exists, so `$size` receives a missing value. The failing expression refers to the corpus join alias (two retrieved examples use `__j1`). The question itself is ill-formed ("more than 3 car makers").
- **Origin:** generation / schema reasoning. **Retrieval signal:** strong. **Corrective action:** F1.
- **Zero-shot:** incorrect (wrong_row_count) — row count 0 != gold 6

RAG MQL:

```json
{"collection":"car_1__car_makers","pipeline":[{"$lookup":{"from":"car_1__model_list","localField":"Id","foreignField":"Maker","as":"__j1"}},{"$unwind":"$__j1"},{"$group":{"_id":"$Id","Maker":{"$first":"$Maker"},"ModelCount":{"$sum":1}}},{"$match":{"$expr":{"$and":[{"$gte":["$ModelCount",2]},{"$gt":[{"$size":"$__j1"},3]}]}}},{"$project":{"_id":0,"Id":"$_id","Maker":"$Maker"}}]}
```

Zero-shot MQL:

```json
{"collection":"car_1__car_makers","pipeline":[{"$lookup":{"from":"car_1__model_list","localField":"Id","foreignField":"Maker","as":"models"}},{"$match":{"models":{"$gte":2}}},{"$group":{"_id":"$Id","Maker":{"$first":"$Maker"}}},{"$sort":{"Maker":1}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.360 | spider_train_06588 | wine_1 | Which 3 wineries produce the most wines made from white grapes? |
| 2 | 1.340 | spider_train_02942 | sakila_1 | Which countries have at least 3 cities? |
| 3 | 1.333 | spider_train_06580 | wine_1 | Which wineries produce at least four wines? |

### 8. spider_dev_0258 (flight_2, hard)

- **Question:** Which airports do not have departing or arriving flights?
- **Atlas error (code 40237):** `The $anyElementTrue accumulator is a unary operator`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `"hasFlights":{"$anyElementTrue":["$__j1","$__j2"]}` inside `$group`
- **Root cause:** C8. Expression operator used as a `$group` accumulator. Confidence: high.
- **Explanation:** `$anyElementTrue` is not an accumulator and takes one argument. A NOT-EXISTS test was needed (`$match` on empty lookup arrays, as zero-shot attempted). No retrieved example contains this operator.
- **Origin:** generation. **Retrieval signal:** none. **Corrective action:** F1, F2.
- **Zero-shot:** incorrect (wrong_column_count) — column count 4 != gold 1

RAG MQL:

```json
{"collection":"flight_2__airports","pipeline":[{"$lookup":{"from":"flight_2__flights","localField":"AirportCode","foreignField":"SourceAirport","as":"__j1"}},{"$lookup":{"from":"flight_2__flights","localField":"AirportCode","foreignField":"DestAirport","as":"__j2"}},{"$unwind":"$__j1"},{"$unwind":"$__j2"},{"$group":{"_id":"$AirportCode","hasFlights":{"$anyElementTrue":["$__j1","$__j2"]}}},{"$match":{"hasFlights":false}},{"$project":{"_id":0,"AirportCode":"$_id","AirportName":1}}]}
```

Zero-shot MQL:

```json
{"collection":"flight_2__airports","pipeline":[{"$lookup":{"from":"flight_2__flights","localField":"AirportCode","foreignField":"SourceAirport","as":"departing_flights"}},{"$lookup":{"from":"flight_2__flights","localField":"AirportCode","foreignField":"DestAirport","as":"arriving_flights"}},{"$match":{"$or":[{"departing_flights":[]},{"arriving_flights":[]}]}},{"$project":{"AirportCode":1,"AirportName":1,"Country":1,"CountryAbbrev":1,"_id":0}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.188 | spider_train_02276 | entrepreneur | Return the names of entrepreneurs do no not have the investor Rachel Elnaugh. |
| 2 | 1.000 | spider_train_01046 | phone_1 | How many models do not have the wifi function? |
| 3 | 0.861 | spider_train_02787 | election | Which counties have two or more delegates? Give me the county names. |

### 9. spider_dev_0364 (cre_Doc_Template_Mgt, medium)

- **Question:** What are the ids and texts of paragraphs in the document titled 'Welcome to NY'?
- **Atlas error (code 2):** `cannot nest $ under $in`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"Document_ID":{"$in":[{"$lookup":...},{"$unwind":"$__j1"},{...}]}}`
- **Root cause:** C9. Pipeline stages nested inside a query `$in` list (subquery emulation). Confidence: high.
- **Explanation:** Stages cannot appear inside a query `$in` list. Zero-shot makes the same mistake on this item without any example, so the subquery habit is the base model's. The RAG output pasted the example's `$lookup`/`$unwind`/`__j1` stages into that list.
- **Origin:** generation (base-model habit). **Retrieval signal:** plausible. **Corrective action:** F1, F2.
- **Zero-shot:** execution error — `cannot nest $ under $in`

RAG MQL:

```json
{"collection":"cre_Doc_Template_Mgt__Paragraphs","pipeline":[{"$match":{"Document_ID":{"$in":[{"$lookup":{"from":"cre_Doc_Template_Mgt__Documents","localField":"Document_ID","foreignField":"Document_ID","as":"__j1"}},{"$unwind":"$__j1"},{"Document_Name":"Welcome to NY"}]}}},{"$project":{"_id":0,"Paragraph_ID":1,"Paragraph_Text":1}}]}
```

Zero-shot MQL:

```json
{"collection":"cre_Doc_Template_Mgt__Paragraphs","pipeline":[{"$match":{"Document_ID":{"$in":[{"$lookup":{"from":"cre_Doc_Template_Mgt__Documents","localField":"Document_ID","foreignField":"Document_ID","as":"document"}},{"$project":{"Document_ID":1,"Document_Name":1}},{"$match":{"Document_Name":"Welcome to NY"}}]}}},{"$project":{"Paragraph_ID":1,"Paragraph_Text":1}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 2.000 | spider_train_00253 | musical | What are the names of actors who have been in the musical titled The Phantom of the Opera? |
| 2 | 1.560 | spider_train_02313 | perpetrator | What are the names of perpetrators in country "China" or "Japan"? |
| 3 | 1.550 | spider_train_02370 | csu_1 | What are the degrees conferred in "San Francisco State University" in 2001. |

### 10. spider_dev_0365 (cre_Doc_Template_Mgt, medium)

- **Question:** Show all paragraph texts for the document "Customer reviews".
- **Atlas error (code 2):** `cannot nest $ under $in`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"Document_ID":{"$in":[{"$lookup":...},{"$unwind":"$__j1"},{...}]}}`
- **Root cause:** C9. Pipeline stages nested inside a query `$in` list (subquery emulation). Confidence: high.
- **Explanation:** Same defect as spider_dev_0364. Zero-shot executed here only because it hard-coded `Document_ID: 1` (wrong answer).
- **Origin:** generation (base-model habit). **Retrieval signal:** plausible. **Corrective action:** F1, F2.
- **Zero-shot:** incorrect (wrong_row_count) — row count 0 != gold 3

RAG MQL:

```json
{"collection":"cre_Doc_Template_Mgt__Paragraphs","pipeline":[{"$match":{"Document_ID":{"$in":[{"$lookup":{"from":"cre_Doc_Template_Mgt__Documents","localField":"Document_ID","foreignField":"Document_ID","as":"__j1"}},{"$unwind":"$__j1"},{"Document_Name":"Customer reviews"}]}}},{"$project":{"_id":0,"Paragraph_Text":1}}]}
```

Zero-shot MQL:

```json
{"collection":"cre_Doc_Template_Mgt__Paragraphs","pipeline":[{"$match":{"Document_ID":1}},{"$project":{"Paragraph_Text":1,"_id":0}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.798 | spider_train_06753 | activity_1 | Show first name and last name for all the students advised by Michael Goodrich. |
| 2 | 1.692 | spider_train_00413 | flight_1 | Show all flight numbers with aircraft Airbus A340-300. |
| 3 | 1.657 | spider_train_01605 | customers_and_invoices | Show invoice dates and order id and details for all invoices. |

### 11. spider_dev_0380 (cre_Doc_Template_Mgt, hard)

- **Question:** What are the ids of documents that contain the paragraph text 'Brazil' and 'Ireland'?
- **Atlas error (code beyond the saved 300 characters):** `unknown top level operator: $regexMatch. If you have a field name that starts with a '$' symbol, consider using $getField or $setField.`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"$match":{"$and":[{"$regexMatch":{...}},{"$regexMatch":{...}}]}}`
- **Root cause:** C1. Aggregation-expression operator used as a query operator in `$match` without `$expr`. Confidence: high.
- **Explanation:** `$regexMatch` is an expression operator; in `$match` it needs `$expr` (or the query form `$regex`). All three retrieved examples use the query form `$regex`; none uses `$regexMatch`, and the corpus contains none. The output does use the corpus alias `__j1`.
- **Origin:** generation. **Retrieval signal:** plausible. **Corrective action:** F1, F2.
- **Zero-shot:** correct — multiset match

RAG MQL:

```json
{"collection":"cre_Doc_Template_Mgt__Documents","pipeline":[{"$lookup":{"from":"cre_Doc_Template_Mgt__Paragraphs","localField":"Document_ID","foreignField":"Document_ID","as":"__j1"}},{"$unwind":"$__j1"},{"$match":{"$and":[{"$regexMatch":{"input":"$__j1.Paragraph_Text","regex":"Brazil","options":"i"}},{"$regexMatch":{"input":"$__j1.Paragraph_Text","regex":"Ireland","options":"i"}}]}},{"$group":{"_id":"$Document_ID"}},{"$project":{"_id":0,"Document_ID":"$_id"}}]}
```

Zero-shot MQL:

```json
{"collection":"cre_Doc_Template_Mgt__Documents","pipeline":[{"$lookup":{"from":"cre_Doc_Template_Mgt__Paragraphs","localField":"Document_ID","foreignField":"Document_ID","as":"paragraphs"}},{"$match":{"paragraphs.Paragraph_Text":{"$in":["Brazil","Ireland"]}}},{"$group":{"_id":"$Document_ID"}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.677 | spider_train_05543 | products_gen_characteristics | What are the descriptions of the categories that products with product descriptions that contain the letter t are in? |
| 2 | 1.500 | spider_train_04530 | document_management | What are the names of documents that contain the substring "CV"? |
| 3 | 1.500 | spider_train_06337 | e_government | What are the names of organizations that contain the word "Party"? |

### 12. spider_dev_0514 (student_transcripts_tracking, easy)

- **Question:** What is the zip code for Port Chelsea?
- **Atlas error (code 8000):** `address_id is not allowed or the syntax is incorrect, see the Atlas documentation for more information`
- **Scorer label:** `failed` / `atlas_tier_restriction`; executed on Atlas: False
- **Failing construct:** the only pipeline element is `{"address_id":1,"line_1":"2294 Grant Square Apt. 235",...}`
- **Root cause:** C4. Pipeline element is a data document, not a stage. Confidence: high for what failed, medium for why.
- **Explanation:** The model emitted a row-like document where a stage belongs. Atlas reports the first key (`address_id`) as an unknown stage. The values are not in the retrieved examples. Zero-shot answered this easy question correctly.
- **Origin:** generation. **Retrieval signal:** none. **Corrective action:** F1.
- **Zero-shot:** correct — multiset match

RAG MQL:

```json
{"collection":"student_transcripts_tracking__Addresses","pipeline":[{"address_id":1,"line_1":"2294 Grant Square Apt. 235","line_2":"Apt. 370","line_3":null,"city":"Port Chelsea","zip_postcode":"148","state_province_county":"Virginia","country":"Iceland","other_address_details":null}]}
```

Zero-shot MQL:

```json
{"collection":"student_transcripts_tracking__Addresses","pipeline":[{"$match":{"city":"Port Chelsea"}},{"$project":{"zip_postcode":1,"_id":0}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.000 | spider_train_02337 | csu_1 | What is the campus fee for San Jose State University in 1996? |
| 2 | 1.000 | spider_train_02339 | csu_1 | What is the campus fee for San Francisco State University in 1996? |
| 3 | 0.988 | spider_train_04224 | cre_Doc_Tracking_DB | What is "the date in location from" and "the date in location to" for the document with name "Robin CV"? |

### 13. spider_dev_0559 (student_transcripts_tracking, easy)

- **Question:** Whose permanent address is different from his or her current address? List his or her first name.
- **Atlas error (code beyond the saved 300 characters):** `unknown top level operator: $ne. If you have a field name that starts with a '$' symbol, consider using $getField or $setField.`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"$match":{"$and":[{"$ne":["$__j1.address_id","$__j2.address_id"]}]}}`
- **Root cause:** C1. Aggregation-expression operator used as a query operator in `$match` without `$expr`. Confidence: high.
- **Explanation:** Field-to-field comparison in array form without `$expr`. Zero-shot wrote the same comparison inside `$expr` and was correct. No retrieved example for this item contains `$expr`; the output uses the corpus aliases `__j1`/`__j2`.
- **Origin:** generation. **Retrieval signal:** plausible. **Corrective action:** F1, F2.
- **Zero-shot:** correct — multiset match

RAG MQL:

```json
{"collection":"student_transcripts_tracking__Students","pipeline":[{"$lookup":{"from":"student_transcripts_tracking__Addresses","localField":"permanent_address_id","foreignField":"address_id","as":"__j1"}},{"$lookup":{"from":"student_transcripts_tracking__Addresses","localField":"current_address_id","foreignField":"address_id","as":"__j2"}},{"$unwind":"$__j1"},{"$unwind":"$__j2"},{"$match":{"$and":[{"$ne":["$__j1.address_id","$__j2.address_id"]}]}},{"$project":{"_id":0,"first_name":"$first_name"}}]}
```

Zero-shot MQL:

```json
{"collection":"student_transcripts_tracking__Students","pipeline":[{"$lookup":{"from":"student_transcripts_tracking__Addresses","localField":"permanent_address_id","foreignField":"address_id","as":"permanent_address"}},{"$lookup":{"from":"student_transcripts_tracking__Addresses","localField":"current_address_id","foreignField":"address_id","as":"current_address"}},{"$match":{"$expr":{"$ne":["$permanent_address.line_1","$current_address.line_1"]}}},{"$project":{"first_name":1,"_id":0}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.000 | spider_train_03803 | e_learning | Which course author teaches the "advanced database" course? Give me his or her login name. |
| 2 | 0.924 | spider_train_01063 | match_season | What are the different positions of players from UCLA or Duke colleges? |
| 3 | 0.911 | spider_train_02952 | sakila_1 | Which film actor (actress) starred the most films? List his or her first name, last name and actor id. |

### 14. spider_dev_0720 (world_1, hard)

- **Question:** What is the name of country that has the shortest life expectancy in Asia?
- **Atlas error (code 8000):** `match is not allowed or the syntax is incorrect, see the Atlas documentation for more information`
- **Scorer label:** `failed` / `atlas_tier_restriction`; executed on Atlas: False
- **Failing construct:** `{"match":...},{"sort":...},{"limit":1},{"project":...}`
- **Root cause:** C3. Pipeline stage name written without `$`. Confidence: high.
- **Explanation:** All four stage names lack `$`. All three retrieved examples are correctly written `$sort`/`$limit`/`$project` pipelines. Zero-shot shows the same defect on three other items, so it is a base-model slip.
- **Origin:** generation (base-model habit). **Retrieval signal:** none. **Corrective action:** F1, F2.
- **Zero-shot:** correct — ordered match (tie-aware)

RAG MQL:

```json
{"collection":"world_1__country","pipeline":[{"match":{"Continent":"Asia"}},{"sort":{"LifeExpectancy":1}},{"limit":1},{"project":{"_id":0,"Name":"$Name"}}]}
```

Zero-shot MQL:

```json
{"collection":"world_1__country","pipeline":[{"$match":{"Continent":"Asia"}},{"$sort":{"LifeExpectancy":1}},{"$limit":1},{"$project":{"Name":1,"_id":0}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 2.000 | spider_train_00119 | bike_1 | What is the id of the trip that has the shortest duration? |
| 2 | 1.698 | spider_train_02277 | entrepreneur | What is the weight of the shortest person? |
| 3 | 1.584 | spider_train_06819 | flight_4 | Which country is the airport that has the highest altitude located in? |

### 15. spider_dev_0737 (world_1, medium)

- **Question:** How many languages are spoken in Aruba?
- **Atlas error (code 8000):** `match is not allowed or the syntax is incorrect, see the Atlas documentation for more information`
- **Scorer label:** `failed` / `atlas_tier_restriction`; executed on Atlas: False
- **Failing construct:** `{"match":...},{"group":...}`
- **Root cause:** C3. Pipeline stage name written without `$`. Confidence: high.
- **Explanation:** Same defect as spider_dev_0720.
- **Origin:** generation (base-model habit). **Retrieval signal:** none. **Corrective action:** F1, F2.
- **Zero-shot:** incorrect (wrong_row_count) — row count 4 != gold 1

RAG MQL:

```json
{"collection":"world_1__countrylanguage","pipeline":[{"match":{"CountryCode":"ABW"}},{"group":{"_id":"$Language","count":{"$sum":1}}}]}
```

Zero-shot MQL:

```json
{"collection":"world_1__countrylanguage","pipeline":[{"$match":{"CountryCode":"ABW"}},{"$group":{"_id":"$Language","count":{"$sum":1}}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 2.000 | spider_train_00130 | bike_1 | How many stations are in Mountain View? |
| 2 | 1.948 | spider_train_00595 | store_1 | How many employees are living in Canada? |
| 3 | 1.948 | spider_train_01325 | college_2 | How many classrooms are not in Lamberton? |

### 16. spider_dev_0738 (world_1, medium)

- **Question:** How many official languages does Afghanistan have?
- **Atlas error (code 8000):** `match is not allowed or the syntax is incorrect, see the Atlas documentation for more information`
- **Scorer label:** `failed` / `atlas_tier_restriction`; executed on Atlas: False
- **Failing construct:** `{"match":...},{"group":...},{"project":...}`
- **Root cause:** C3. Pipeline stage name written without `$`. Confidence: high.
- **Explanation:** Same defect as spider_dev_0720.
- **Origin:** generation (base-model habit). **Retrieval signal:** none. **Corrective action:** F1, F2.
- **Zero-shot:** incorrect (wrong_row_count) — row count 0 != gold 1

RAG MQL:

```json
{"collection":"world_1__countrylanguage","pipeline":[{"match":{"CountryCode":"AFG"}},{"group":{"_id":null,"officialLanguagesCount":{"$sum":1}}},{"project":{"officialLanguagesCount":1,"_id":0}}]}
```

Zero-shot MQL:

```json
{"collection":"world_1__countrylanguage","pipeline":[{"$match":{"CountryCode":"AFG"}},{"$group":{"_id":"$Language","count":{"$sum":1}}},{"$match":{"IsOfficial":"T"}},{"$project":{"_id":0,"count":1}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 2.000 | spider_train_03688 | baseball_1 | How many parks does Atlanta city have? |
| 2 | 1.955 | spider_train_01334 | college_2 | How many rooms does the Lamberton building have? |
| 3 | 1.887 | spider_train_00277 | twitter_1 | How many followers does each user have? |

### 17. spider_dev_0739 (world_1, medium)

- **Question:** How many official languages are spoken in Afghanistan?
- **Atlas error (code 8000):** `match is not allowed or the syntax is incorrect, see the Atlas documentation for more information`
- **Scorer label:** `failed` / `atlas_tier_restriction`; executed on Atlas: False
- **Failing construct:** `{"match":...},{"group":...}`
- **Root cause:** C3. Pipeline stage name written without `$`. Confidence: high.
- **Explanation:** Same defect as spider_dev_0720. Same retrieved examples as spider_dev_0737.
- **Origin:** generation (base-model habit). **Retrieval signal:** none. **Corrective action:** F1, F2.
- **Zero-shot:** incorrect (wrong_row_count) — row count 0 != gold 1

RAG MQL:

```json
{"collection":"world_1__countrylanguage","pipeline":[{"match":{"CountryCode":"AFG"}},{"group":{"_id":"$Language","count":{"$sum":1}}}]}
```

Zero-shot MQL:

```json
{"collection":"world_1__countrylanguage","pipeline":[{"$match":{"CountryCode":"AFG"}},{"$group":{"_id":"$Language","count":{"$sum":1}}},{"$match":{"IsOfficial":"T"}},{"$project":{"_id":0,"count":1}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 2.000 | spider_train_00130 | bike_1 | How many stations are in Mountain View? |
| 2 | 1.948 | spider_train_00595 | store_1 | How many employees are living in Canada? |
| 3 | 1.948 | spider_train_01325 | college_2 | How many classrooms are not in Lamberton? |

### 18. spider_dev_0755 (world_1, extra)

- **Question:** Which countries have either English or Dutch as an official language?
- **Atlas error (code beyond the saved 300 characters):** `PlanExecutor error during aggregation :: caused by :: error while multiplanner was selecting best plan :: caused by :: $in requires an array as a second argument, found: string`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"$in":["$$code","$CountryCode"]}` inside the `$lookup` sub-pipeline
- **Root cause:** C6. `$lookup` with `let`/`pipeline`: `$in` given a scalar instead of an array. Confidence: high.
- **Explanation:** `$CountryCode` is a string, not an array; `$eq` was meant. No retrieved example and no corpus record uses a `let`/`pipeline` lookup.
- **Origin:** generation. **Retrieval signal:** none. **Corrective action:** F1.
- **Zero-shot:** incorrect (wrong_row_count) — row count 233 != gold 48

RAG MQL:

```json
{"collection":"world_1__country","pipeline":[{"$lookup":{"from":"world_1__countrylanguage","let":{"code":"$Code"},"pipeline":[{"$match":{"$expr":{"$in":["$$code","$CountryCode"]}}},{"$match":{"$or":[{"IsOfficial":"T"},{"IsOfficial":"F"}]}},{"$group":{"_id":"$Language"}}],"as":"languages"}},{"$unwind":"$languages"},{"$group":{"_id":"$Code","languages":{"$addToSet":"$languages._id"}}},{"$match":{"languages":{"$in":["English","Dutch"]}}}]}
```

Zero-shot MQL:

```json
{"collection":"world_1__country","pipeline":[{"$lookup":{"from":"world_1__countrylanguage","localField":"Code","foreignField":"CountryCode","as":"languages"}},{"$unwind":"$languages"},{"$match":{"$or":[{"languages.IsOfficial":"T"},{"languages.IsOfficial":"F"}]}},{"$group":{"_id":"$Name"}},{"$sort":{"_id":1}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.000 | spider_train_02284 | entrepreneur | Return the dates of birth for entrepreneurs who have either the investor Simon Woodroffe or Peter Jones. |
| 2 | 0.966 | spider_train_03287 | college_1 | How many professors who has a either Ph.D. or MA degree? |
| 3 | 0.952 | spider_train_04580 | company_office | How many companies are in either "Banking" industry or "Conglomerate" industry? |

### 19. spider_dev_0839 (orchestra, medium)

- **Question:** What is the name of the conductor who has worked the greatest number of years?
- **Atlas error (code 40237):** `The $subtract accumulator is a unary operator`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `"years_worked":{"$subtract":["$Year_of_Work",{"$year":{"$toDate":"$Date"}}]}` inside `$group`
- **Root cause:** C8. Expression operator used as a `$group` accumulator. Confidence: high.
- **Explanation:** `$subtract` is not an accumulator. It also refers to `Date`, which is not a field of the conductor collection. Zero-shot makes the same accumulator mistake on spider_dev_0837. The retrieved examples are plain counts.
- **Origin:** generation / schema interpretation. **Retrieval signal:** none. **Corrective action:** F1, F2.
- **Zero-shot:** incorrect (wrong_column_count) — column count 2 != gold 1

RAG MQL:

```json
{"collection":"orchestra__conductor","pipeline":[{"$group":{"_id":"$Name","years_worked":{"$subtract":["$Year_of_Work",{"$year":{"$toDate":"$Date"}}]}}},{"$sort":{"years_worked":-1}},{"$limit":1}]}
```

Zero-shot MQL:

```json
{"collection":"orchestra__conductor","pipeline":[{"$group":{"_id":"$Name","years_worked":{"$sum":{"$subtract":["$Year_of_Work",1900]}}}},{"$sort":{"years_worked":-1}},{"$limit":1}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.608 | spider_train_02448 | movie_1 | What is the total number of ratings that has more than 3 stars? |
| 2 | 1.519 | spider_train_00382 | flight_1 | What is the number of flights? |
| 3 | 1.519 | spider_train_03999 | ship_mission | What is the number of ships? |

### 20. spider_dev_0898 (network_1, hard)

- **Question:** Show the ids of high schoolers who have friends and are also liked by someone else.
- **Atlas error (code beyond the saved 300 characters):** `unknown top level operator: $gt. If you have a field name that starts with a '$' symbol, consider using $getField or $setField.`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"$match":{"$and":[{"$gt":[{"$size":"$__j1"},0]},...]}}`
- **Root cause:** C1. Aggregation-expression operator used as a query operator in `$match` without `$expr`. Confidence: high.
- **Explanation:** Array-form comparison without `$expr`. No retrieved example for this item contains `$expr` or `$size`; scores are low (1.00, 0.80, 0.75). The output uses the corpus aliases `__j1`..`__j3`.
- **Origin:** generation. **Retrieval signal:** plausible. **Corrective action:** F1, F2.
- **Zero-shot:** incorrect (wrong_values) — some predicted column matches no gold column

RAG MQL:

```json
{"collection":"network_1__Highschooler","pipeline":[{"$lookup":{"from":"network_1__Friend","localField":"ID","foreignField":"student_id","as":"__j1"}},{"$lookup":{"from":"network_1__Friend","localField":"ID","foreignField":"friend_id","as":"__j2"}},{"$lookup":{"from":"network_1__Likes","localField":"ID","foreignField":"student_id","as":"__j3"}},{"$match":{"$and":[{"$gt":[{"$size":"$__j1"},0]},{"$gt":[{"$size":"$__j2"},0]},{"$gt":[{"$size":"$__j3"},0]}]}},{"$project":{"_id":0,"ID":1}}]}
```

Zero-shot MQL:

```json
{"collection":"network_1__Highschooler","pipeline":[{"$lookup":{"from":"network_1__Friend","localField":"ID","foreignField":"student_id","as":"friends"}},{"$lookup":{"from":"network_1__Likes","localField":"ID","foreignField":"student_id","as":"likes"}},{"$match":{"$and":[{"friends":{"$ne":[]}},{"likes":{"$ne":[]}}]}},{"$project":{"_id":0,"ID":1}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.000 | spider_train_04999 | soccer_2 | What are the names of all the players who received a yes during tryouts, and also what are the names of their colleges? |
| 2 | 0.802 | spider_train_04931 | store_product | What are the products with the maximum page size A4 that also have a pages per minute color smaller than 5? |
| 3 | 0.754 | spider_train_05991 | game_1 | Show ids for all students who have advisor 1121. |

### 21. spider_dev_0899 (network_1, hard)

- **Question:** What are the ids of students who both have friends and are liked?
- **Atlas error (code beyond the saved 300 characters):** `unknown top level operator: $gte. If you have a field name that starts with a '$' symbol, consider using $getField or $setField.`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"$match":{"$and":[{"$gte":[{"$size":"$__j1"},1]},{"$gte":[{"$size":"$__j2"},1]}]}}`
- **Root cause:** C1. Aggregation-expression operator used as a query operator in `$match` without `$expr`. Confidence: high.
- **Explanation:** Same defect as spider_dev_0898.
- **Origin:** generation. **Retrieval signal:** plausible. **Corrective action:** F1, F2.
- **Zero-shot:** incorrect (wrong_row_count) — row count 0 != gold 8

RAG MQL:

```json
{"collection":"network_1__Highschooler","pipeline":[{"$lookup":{"from":"network_1__Friend","localField":"ID","foreignField":"student_id","as":"__j1"}},{"$lookup":{"from":"network_1__Likes","localField":"ID","foreignField":"student_id","as":"__j2"}},{"$match":{"$and":[{"$gte":[{"$size":"$__j1"},1]},{"$gte":[{"$size":"$__j2"},1]}]}},{"$project":{"_id":0,"ID":1}}]}
```

Zero-shot MQL:

```json
{"collection":"network_1__Highschooler","pipeline":[{"$lookup":{"from":"network_1__Friend","localField":"ID","foreignField":"student_id","as":"friends"}},{"$lookup":{"from":"network_1__Likes","localField":"ID","foreignField":"student_id","as":"likes"}},{"$match":{"friends.length":{"$gt":0},"likes.length":{"$gt":0}}},{"$project":{"_id":0,"ID":1}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.000 | spider_train_00688 | customers_card_transactions | What are the full names of customers who have accounts? |
| 2 | 0.961 | spider_train_03589 | music_1 | What are the names of artists who are Male and are from UK? |
| 3 | 0.938 | spider_train_05485 | voter_2 | What are the first names and last names of the students who are 18 years old and have vice president votes. |

### 22. spider_dev_0900 (network_1, hard)

- **Question:** Show name of all students who have some friends and also are liked by someone else.
- **Atlas error (code beyond the saved 300 characters):** `unknown top level operator: $gt. If you have a field name that starts with a '$' symbol, consider using $getField or $setField.`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"$match":{"$or":[{"$gt":[{"$size":"$__j1"},0]},{"$gt":[{"$size":"$__j2"},0]}]}}`
- **Root cause:** C1. Aggregation-expression operator used as a query operator in `$match` without `$expr`. Confidence: high.
- **Explanation:** Same defect as spider_dev_0898.
- **Origin:** generation. **Retrieval signal:** plausible. **Corrective action:** F1, F2.
- **Zero-shot:** incorrect (wrong_values) — some predicted column matches no gold column

RAG MQL:

```json
{"collection":"network_1__Highschooler","pipeline":[{"$lookup":{"from":"network_1__Friend","localField":"ID","foreignField":"student_id","as":"__j1"}},{"$lookup":{"from":"network_1__Likes","localField":"ID","foreignField":"student_id","as":"__j2"}},{"$match":{"$or":[{"$gt":[{"$size":"$__j1"},0]},{"$gt":[{"$size":"$__j2"},0]}]}},{"$project":{"_id":0,"name":1}}]}
```

Zero-shot MQL:

```json
{"collection":"network_1__Highschooler","pipeline":[{"$lookup":{"from":"network_1__Friend","localField":"ID","foreignField":"student_id","as":"friends"}},{"$lookup":{"from":"network_1__Likes","localField":"ID","foreignField":"student_id","as":"likes"}},{"$match":{"$and":[{"friends":{"$ne":[]}},{"likes":{"$ne":[]}}]}},{"$project":{"name":1,"_id":0}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.000 | spider_train_01468 | college_2 | Find the names of all instructors who have taught some course and the course_id. |
| 2 | 0.966 | spider_train_03309 | college_1 | Find the first names of all instructors who have taught some course and the course code. |
| 3 | 0.950 | spider_train_02132 | company_1 | find all dependent names who have a spouse relation with some employee. |

### 23. spider_dev_0901 (network_1, hard)

- **Question:** What are the names of high schoolers who both have friends and are liked?
- **Atlas error (code beyond the saved 300 characters):** `unknown top level operator: $gt. If you have a field name that starts with a '$' symbol, consider using $getField or $setField.`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"$match":{"$and":[{"$gt":[{"$size":"$__j1"},0]},{"$gt":[{"$size":"$__j2"},0]}]}}`
- **Root cause:** C1. Aggregation-expression operator used as a query operator in `$match` without `$expr`. Confidence: high.
- **Explanation:** Same defect as spider_dev_0898. Same retrieved examples as spider_dev_0899.
- **Origin:** generation. **Retrieval signal:** plausible. **Corrective action:** F1, F2.
- **Zero-shot:** incorrect (wrong_values) — some predicted column matches no gold column

RAG MQL:

```json
{"collection":"network_1__Highschooler","pipeline":[{"$lookup":{"from":"network_1__Friend","localField":"ID","foreignField":"student_id","as":"__j1"}},{"$lookup":{"from":"network_1__Likes","localField":"ID","foreignField":"student_id","as":"__j2"}},{"$match":{"$and":[{"$gt":[{"$size":"$__j1"},0]},{"$gt":[{"$size":"$__j2"},0]}]}},{"$project":{"_id":0,"name":1}}]}
```

Zero-shot MQL:

```json
{"collection":"network_1__Highschooler","pipeline":[{"$lookup":{"from":"network_1__Friend","localField":"ID","foreignField":"student_id","as":"friends"}},{"$lookup":{"from":"network_1__Likes","localField":"ID","foreignField":"student_id","as":"likes"}},{"$match":{"$and":[{"friends":{"$ne":[]}},{"likes":{"$ne":[]}}]}},{"$project":{"name":1,"_id":0}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.000 | spider_train_00688 | customers_card_transactions | What are the full names of customers who have accounts? |
| 2 | 0.961 | spider_train_03589 | music_1 | What are the names of artists who are Male and are from UK? |
| 3 | 0.938 | spider_train_05485 | voter_2 | What are the first names and last names of the students who are 18 years old and have vice president votes. |

### 24. spider_dev_0918 (dog_kennels, hard)

- **Question:** Which states have both owners and professionals living there?
- **Atlas error (code beyond the saved 300 characters):** `unknown top level operator: $gte. If you have a field name that starts with a '$' symbol, consider using $getField or $setField.`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"$match":{"$and":[{"$gte":["$owners",1]},{"$gte":["$professionals",1]}]}}`
- **Root cause:** C1. Aggregation-expression operator used as a query operator in `$match` without `$expr`. Confidence: high.
- **Explanation:** Array-form comparison without `$expr`. Retrieved example spider_train_06656 (rank 1) filters a grouped count with the array form under `$expr`; the output keeps the form and drops `$expr`. Zero-shot used the query form `{"owners":{"$gt":0}}` and executed.
- **Origin:** generation (incomplete imitation of a retrieved pattern). **Retrieval signal:** strong. **Corrective action:** F1, F2, F3.
- **Zero-shot:** incorrect (wrong_row_count) — row count 14 != gold 3

RAG MQL:

```json
{"collection":"dog_kennels__Owners","pipeline":[{"$lookup":{"from":"dog_kennels__Professionals","let":{"owner_id":"$owner_id"},"pipeline":[{"$match":{"$expr":{"$eq":["$$owner_id","$owner_id"]}}}],"as":"professionals"}},{"$unwind":"$professionals"},{"$group":{"_id":"$state","owners":{"$sum":1},"professionals":{"$sum":1}}},{"$match":{"$and":[{"$gte":["$owners",1]},{"$gte":["$professionals",1]}]}},{"$project":{"_id":0,"state":"$_id"}}]}
```

Zero-shot MQL:

```json
{"collection":"dog_kennels__Owners","pipeline":[{"$group":{"_id":"$state","owners":{"$sum":1},"professionals":{"$sum":1}}},{"$match":{"owners":{"$gt":0},"professionals":{"$gt":0}}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.200 | spider_train_06656 | driving_school | List the states which have between 2 to 4 staffs living there. |
| 2 | 0.919 | spider_train_02773 | election | Which committees have delegates from the Democratic party? |
| 3 | 0.901 | spider_train_00334 | product_catalog | Which attribute definitions have attribute value 0? Give me the attribute name and attribute ID. |

### 25. spider_dev_0922 (dog_kennels, extra)

- **Question:** Which professionals live in the state of Indiana or have done treatment on more than 2 treatments? List his or her id, last name and cell phone.
- **Atlas error (code 17124):** `PlanExecutor error during aggregation :: caused by :: The argument to $size must be an array, but was of type: missing`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"$expr":{"$gt":[{"$size":"$treatments"},2]}}`
- **Root cause:** C7. `$size` applied to a field that does not exist. Confidence: high.
- **Explanation:** `Professionals` has no `treatments` field; a join to `Treatments` was needed. Zero-shot invents the same field (`{"treatments":{"$gt":2}}`), which silently matches nothing and is scored incorrect. The RAG arm wrapped the same invented field in `$expr`/`$size`, which fails loudly. All three retrieved examples use `$expr`.
- **Origin:** schema interpretation (same in both arms). **Retrieval signal:** plausible. **Corrective action:** F1.
- **Zero-shot:** incorrect (wrong_row_count) — row count 2 != gold 3

RAG MQL:

```json
{"collection":"dog_kennels__Professionals","pipeline":[{"$match":{"$or":[{"state":"Indiana"},{"$expr":{"$gt":[{"$size":"$treatments"},2]}}]}},{"$project":{"_id":0,"professional_id":1,"last_name":1,"cell_number":1}}]}
```

Zero-shot MQL:

```json
{"collection":"dog_kennels__Professionals","pipeline":[{"$match":{"$or":[{"state":"Indiana"},{"treatments":{"$gt":2}}]}},{"$project":{"_id":0,"professional_id":1,"last_name":1,"cell_number":1}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.600 | spider_train_02954 | sakila_1 | Which film actors (actresses) played a role in more than 30 films? List his or her first name and last name. |
| 2 | 1.160 | spider_train_02787 | election | Which counties have two or more delegates? Give me the county names. |
| 3 | 1.110 | spider_train_03132 | assets_maintenance | Which parts have more than 2 faults? Show the part name and id. |

### 26. spider_dev_0923 (dog_kennels, extra)

- **Question:** Find the id, last name and cell phone of the professionals who live in the state of Indiana or have performed more than two treatments.
- **Atlas error (code 17124):** `PlanExecutor error during aggregation :: caused by :: The argument to $size must be an array, but was of type: missing`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"$expr":{"$gt":[{"$size":"$$ROOT.Treatments"},2]}}`
- **Root cause:** C7. `$size` applied to a field that does not exist. Confidence: high.
- **Explanation:** Same defect as spider_dev_0922 (invented field `Treatments`).
- **Origin:** schema interpretation (same in both arms). **Retrieval signal:** plausible. **Corrective action:** F1.
- **Zero-shot:** incorrect (wrong_row_count) — row count 2 != gold 3

RAG MQL:

```json
{"collection":"dog_kennels__Professionals","pipeline":[{"$match":{"$or":[{"state":"Indiana"},{"$expr":{"$gt":[{"$size":"$$ROOT.Treatments"},2]}}]}},{"$project":{"_id":0,"professional_id":1,"last_name":1,"cell_number":1}}]}
```

Zero-shot MQL:

```json
{"collection":"dog_kennels__Professionals","pipeline":[{"$match":{"$or":[{"state":"Indiana"},{"treatments":{"$gt":2}}]}},{"$project":{"_id":0,"professional_id":1,"last_name":1,"cell_number":1}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.400 | spider_train_02292 | entrepreneur | Return the investors who have invested in two or more entrepreneurs. |
| 2 | 1.355 | spider_train_03062 | loan_1 | Find the name of customers who have more than one loan. |
| 3 | 1.333 | spider_train_02787 | election | Which counties have two or more delegates? Give me the county names. |

### 27. spider_dev_0932 (dog_kennels, medium)

- **Question:** Which professionals have done at least two treatments? List the professional's id, role, and first name.
- **Atlas error (code 40234):** `The field 'role' must be an accumulator object`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `"role":"$role_code","first_name":"$first_name"` inside `$group`
- **Root cause:** C2. `$group` output field that is not an accumulator. Confidence: high.
- **Explanation:** Non-accumulator fields in `$group`. Two retrieved examples show the correct `{"$first":...}` form, which the output did not follow; the rest of the pipeline (join alias `__j1`, `$expr` HAVING) does follow them.
- **Origin:** generation. **Retrieval signal:** plausible. **Corrective action:** F1, F2.
- **Zero-shot:** incorrect (wrong_row_count) — row count 0 != gold 6

RAG MQL:

```json
{"collection":"dog_kennels__Professionals","pipeline":[{"$lookup":{"from":"dog_kennels__Treatments","localField":"professional_id","foreignField":"professional_id","as":"__j1"}},{"$unwind":"$__j1"},{"$group":{"_id":"$professional_id","role":"$role_code","first_name":"$first_name","treatment_count":{"$sum":1}}},{"$match":{"$expr":{"$gte":["$treatment_count",2]}}},{"$project":{"_id":0,"professional_id":1,"role":1,"first_name":1}}]}
```

Zero-shot MQL:

```json
{"collection":"dog_kennels__Professionals","pipeline":[{"$lookup":{"from":"dog_kennels__Treatments","localField":"professional_id","foreignField":"professional_id","as":"treatments"}},{"$match":{"treatments":{"$gte":2}}},{"$project":{"_id":0,"professional_id":1,"role_code":1,"first_name":1}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.333 | spider_train_02930 | sakila_1 | Which districts have at least two addresses? |
| 2 | 1.242 | spider_train_02942 | sakila_1 | Which countries have at least 3 cities? |
| 3 | 1.222 | spider_train_02786 | election | Show the names of counties that have at least two delegates. |

### 28. spider_dev_0974 (dog_kennels, hard)

- **Question:** How many dogs have an age below the average?
- **Atlas error (code 28651):** `PlanExecutor error during aggregation :: caused by :: input to $filter must be an array not object`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"$filter":{"input":"$$ROOT",...}}` after `{"$group":{"_id":null,...}}`
- **Root cause:** C11. `$filter` applied to a document (`$$ROOT`) instead of an array. Confidence: high.
- **Explanation:** `$$ROOT` is a document, and after the `$group` the individual dogs are gone. Retrieved example spider_train_00689 contains `{"$size":{"$filter":{"input":"$a0",...}}}`; the output reuses `$filter` + `$size`, which zero-shot never uses, on the wrong input.
- **Origin:** generation (misapplied retrieved pattern). **Retrieval signal:** strong. **Corrective action:** F1.
- **Zero-shot:** incorrect (wrong_row_count) — row count 0 != gold 1

RAG MQL:

```json
{"collection":"dog_kennels__Dogs","pipeline":[{"$group":{"_id":null,"averageAge":{"$avg":"$age"}}},{"$addFields":{"dogsBelowAverage":{"$filter":{"input":"$$ROOT","as":"dog","cond":{"$lt":["$$dog.age","$averageAge"]}}}}},{"$project":{"_id":0,"count":{"$size":"$dogsBelowAverage"}}}]}
```

Zero-shot MQL:

```json
{"collection":"dog_kennels__Dogs","pipeline":[{"$group":{"_id":null,"count":{"$sum":1},"averageAge":{"$avg":"$age"}}},{"$match":{"age":{"$lt":"$averageAge"}}},{"$project":{"_id":0,"count":1}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.370 | spider_train_02946 | sakila_1 | How many customers have an active value of 1? |
| 2 | 1.250 | spider_train_00689 | customers_card_transactions | How many customers have an account? |
| 3 | 1.211 | spider_train_05672 | dorm_1 | How many female students (sex is F) whose age is below 25? |

### 29. spider_dev_1021 (singer, medium)

- **Question:** what are the different names of the singers that have sales more than 300000?
- **Atlas error (code 28818):** `path option to $unwind stage should be prefixed with a '$': __j1`
- **Scorer label:** `failed` / `execution_error`; executed on Atlas: False
- **Failing construct:** `{"$unwind":"__j1"}`
- **Root cause:** C5. `$unwind` path without the `$` prefix. Confidence: high.
- **Explanation:** Same defect as spider_dev_0044. Zero-shot was correct here, so this is a real loss.
- **Origin:** generation (miscopy of a retrieved pattern). **Retrieval signal:** strong. **Corrective action:** F1, F2.
- **Zero-shot:** correct — multiset match

RAG MQL:

```json
{"collection":"singer__song","pipeline":[{"$match":{"Sales":{"$gt":300000}}},{"$lookup":{"from":"singer__singer","localField":"Singer_ID","foreignField":"Singer_ID","as":"__j1"}},{"$unwind":"__j1"},{"$group":{"_id":"$__j1.Name"}},{"$project":{"_id":0,"Name":"$_id"}}]}
```

Zero-shot MQL:

```json
{"collection":"singer__singer","pipeline":[{"$lookup":{"from":"singer__song","localField":"Singer_ID","foreignField":"Singer_ID","as":"songs"}},{"$match":{"songs.Sales":{"$gt":300000}}},{"$group":{"_id":"$Name"}},{"$project":{"_id":0,"Name":"$_id"}}]}
```

Retrieved examples:

| Rank | Score | Example id | Database | Question |
|---:|---:|---|---|---|
| 1 | 1.418 | spider_train_04015 | ship_mission | What are the nations that have more than two ships? |
| 2 | 1.392 | spider_train_00628 | store_1 | What are the names of the albums that have more than 10 tracks? |
| 3 | 1.385 | spider_train_02079 | party_people | find the event names that have more than 2 records. |

## 5. Retrieval influence analysis

### What retrieval measurably changes (all 892 items per arm)

| Construct in the generated MQL | Zero-shot outputs | RAG outputs | RAG: correct | RAG: execution errors |
|---|---:|---:|---:|---:|
| Join alias `__jN` (corpus idiom) | 0 | 203 | 58 (28.6%) | 14 |
| `$unwind` | 172 | 323 | 88 (27.2%) | 16 |
| `$expr` | 6 | 69 | 24 (34.8%) | 8 |
| `$lookup` with `let`/`pipeline` | 0 | 26 | 8 (30.8%) | 3 |
| Aliases `a0`/`g0` (corpus idiom) | 0 | 20 | 7 (35.0%) | 0 |
| Bare expression operator in `$match` | 0 | 8 | 0 | 8 |
| Stage name without `$` | 3 | 5 | 0 | 5 |

The "stage without `$`" row for RAG counts the four C3 cases and the C4 case (spider_dev_0514).

Corpus v2.1 (3,445 records): 835 use `__jN`, 826 `$unwind`, 183 `$expr`, 196 `$size`/`$filter`, 114 `$regex`. **0** contain a bare expression operator in `$match`, a stage without `$`, a `let`/`pipeline` lookup, a user `$$` variable or `$regexMatch`.

Reading: the examples pull the model towards join-and-unwind pipelines and expression-style filters. That shift is part of why the arm gains 42 items overall, and it is also where 16 of the 29 errors sit (`$unwind` present). The examples are correct; the model's reproduction of them is not always.

### Per-case classification

**Directly supported (6 cases: 0044, 0171, 0175, 0918, 0974, 1021).** The failing token or expression contains a construct that (a) is present in that item's retrieved examples, (b) never appears in any zero-shot output, and (c) is reproduced incompletely: `"$__j1"` copied as `"__j1"` (0044, 1021); the array-form comparison of a `$expr` example copied without `$expr` (0171, 0918); the join alias used after it was grouped away (0175); `$filter`+`$size` from an example applied to `$$ROOT` (0974).

**Plausible association (13 cases: 0090, 0093, 0364, 0365, 0380, 0559, 0898, 0899, 0900, 0901, 0922, 0923, 0932).** The output carries corpus idioms (`__jN`, `$expr`), but the failing construct is not in that item's three examples, or the same mistake also occurs without retrieval. For the six C1 cases here the association rests on the aggregate fact that the pattern exists only in the RAG arm, not on the item's own examples.

**No identifiable relationship (10 cases: 0042, 0168, 0258, 0514, 0720, 0737, 0738, 0739, 0755, 0839).** The failing construct appears in no retrieved example and nowhere in the corpus (`let`/`pipeline` lookups, `$$` variables, `$anyElementTrue`), or it is a slip that zero-shot makes too (stage names without `$`, `$subtract` as accumulator).

### Retrieval rank and score

All 29 cases received three examples at ranks 1 to 3 (no rule exclusion, no budget skip). Mean top-1 score: 1.434 for the 29 error cases against 1.541 for all 892 items. 8 of the 29 have a top-1 score of 1.00 or lower. This is descriptive only; it is too small a sample to conclude that weak retrieval causes errors.

### What is not claimed

- No statement here says that a specific example *caused* an error. The model's internals were not inspected; the classification is based on textual correspondence and on the contrast with zero-shot.
- The presence of an example in the prompt is never used as evidence on its own.

## 6. Evidence limitations

- **Error text is truncated.** The scorer stores the exception type plus the first 300 characters of the message. That includes the server `errmsg` and, for 19 of 29, the error code (for the other 10 the code lies beyond the saved 300 characters). No stack trace is stored (MongoDB does not return one), and the failing stage index is not reported by the server; the failing construct was identified by reading the pipeline against the message. In every case exactly one construct matches the message.
- **Five cases carry the scorer label `atlas_tier_restriction`** (Atlas code 8000: spider_dev_0514, 0720, 0737, 0738, 0739). The saved message (`match is not allowed or the syntax is incorrect`) shows these are malformed stage names, not tier restrictions. The label is a naming issue in the scorer; the outcome category (execution error) is right.
- **Schemas** were read from the package's `atlas_collections.json` (collection and field names). Field values on Atlas were not inspected, as Atlas was not contacted.
- **One scoring run.** These are parse-time or first-document failures and should be deterministic, but this was not re-tested.
- **Causation.** Retrieval influence is associative (see section 5).
- **spider_dev_0514:** what failed is certain; why the model emitted a row-like document is not known.

## 7. Three prioritized recommendations

None of these may be tuned on the 892 evaluation questions. Each must be developed and checked on internal data (section 8) and then measured as a new, pre-registered experiment.

| Rank | Recommendation | Type | Cases addressed (of 29) | Likelihood of improving execution reliability | Risk to currently correct predictions | Complexity |
|---:|---|---|---|---|---|---|
| 1 | **Validate-and-retry at generation time.** Before a prediction is accepted, check the pipeline against a local MongoDB loaded with the schema only (empty collections) or with a few synthetic rows; on rejection, give the model the server message once and regenerate. | Prompt / generation | 23 detectable on empty collections (all parse-time categories); up to 29 if a few rows per collection are loaded | High: the errors are deterministic and the server message names the fault | Low: predictions that already execute are never touched. A retry can still return a wrong answer, so accuracy gain is at most the retried items | Medium: needs a MongoDB process inside the Kaggle session, or the retry step run locally before export; needs its own leakage review (no evaluation data values) |
| 2 | **Short rule block in the prompt** (both arms, to keep the comparison fair): field-to-field and array-form comparisons go inside `$expr`; every non-`_id` field of `$group` is an accumulator; stage names and `$unwind` paths start with `$`; no stages inside `$in`. | Prompt / generation | 21 (C1 8, C2 3, C8 2, C3 4, C5 2, C9 2); also 8 of zero-shot's 13 errors are of these kinds | Medium: targets the most frequent families, but small models follow such rules imperfectly | Medium: any prompt change can move currently correct outputs in both arms; must be measured, not assumed | Low |
| 3 | **Change how examples are displayed, not the corpus.** Render retrieved examples in a plainer form for the prompt (readable join aliases, the `$expr` wrapper made explicit and never elided, the count wrapper simplified), with every display form re-verified against the stored label. | Retrieval (presentation) | 6 strong cases directly; up to 19 including the plausible ones | Low to medium: unproven; it removes the opportunity for incomplete copying | Highest of the three: the same imitation drives the +42 gain (58 correct outputs use the corpus alias) | Medium to high: a display transform plus equivalence checks on all 3,445 records |

By area:

- **Retrieval improvements:** recommendation 3 only. No evidence supports changing the retriever, K, weights or the corpus contents; the examples were correct, and the error cases' top-1 scores (mean 1.43) are close to the overall mean (1.54).
- **Prompt or generation improvements:** recommendations 1 and 2.
- **Converter or corpus improvements:** none. No corpus record contains a failing construct, and no error traces to a wrong label. A converter rewrite is not supported by this evidence.
- **Scoring or execution infrastructure (reporting only, no change to outcomes or metric):** store the full server message and the error code as separate fields; rename the code-8000 detail label, which here marks malformed stages rather than tier limits. Both would apply to future runs only; MQL-0008 outputs stay as they are.

**Do not chase execution errors for their own sake.** Removing all 29 would recover at most the 6 items zero-shot had right, plus whatever share of the other 23 a retry answers correctly. The RAG arm is ahead by 42 with these errors included.

## 8. Suggested internal regression tests

All on internal data only: the approved corpus, the spider-pool-v3 validation split and the local MongoDB 8.0.34. Local results are label-validation or tooling checks, never reported model metrics.

1. **Negative fixtures, one per category C1 to C11.** Synthetic pipelines on a toy schema reproducing each construct (not the 29 evaluation pipelines themselves); the validator must reject each with the expected error family.
2. **No false positives on the corpus.** All 3,445 approved-corpus pipelines must pass the validator on their own databases. Any rejection is a validator bug.
3. **Parse-time versus data-time split.** On empty collections the validator must reject C1, C2, C3, C4, C5, C8, C9, C10 and accept-then-flag C6, C7, C11 only when rows are present; the test documents which categories need rows.
4. **Retry loop.** With a mocked generator: a rejected pipeline triggers exactly one retry that includes the server message; an accepted pipeline is returned byte-identical; a second failure is recorded, not hidden.
5. **Prompt rule block.** Prompt hashes for both arms change together; arm A and arm B prompts still differ only by the reference block.
6. **Display transform (if recommendation 3 is pursued).** For every corpus record, the displayed MQL executes on the local database and returns the same result as the stored label.
7. **Scorer reporting.** Unit test that code-8000 messages of the form `<name> is not allowed or the syntax is incorrect` are labelled as malformed stages, and that the full message is stored; outcome categories unchanged for all existing fixtures.
8. **Held-out check before any new evaluation.** Measure execution-error rate with and without each change on a development set drawn from Spider-train databases, with the analysis plan registered first.

