# Manual Drawing Evaluation

## Evidence and method

Recorded on November 5, 2026. This analysis transcribes the labels and confidence scores visible in 25 screenshots supplied by the user. It does not rerun the model. Intended categories are identified from the filenames and visible drawings. Trial numbers follow the filenames.

The set contains five drawings each of cat, car, tree, umbrella, and clock. Confidence values are copied as displayed, rounded to one decimal place. Averages below are calculated from those displayed values.

## Results

| Evaluation | Images | Top-1 accuracy | Top-3 accuracy |
| --- | ---: | ---: | ---: |
| Supplied manual drawing screenshots | 25 | 100.00% (25/25) | 100.00% (25/25) |
| Previously reported held-out Quick, Draw! test | 1,500 | 95.13% (1,427/1,500) | 99.47% |

The held-out results above come from the earlier evaluate.py output supplied by the user; they were not remeasured here. The two evaluations use different samples, so their percentages should not be interpreted as a measured improvement.

| Category | Top-1 correct | Top-3 correct | Mean displayed top-1 confidence | Range |
| --- | ---: | ---: | ---: | ---: |
| cat | 5/5 | 5/5 | 64.78% | 55.8-78.4% |
| car | 5/5 | 5/5 | 58.90% | 47.2-83.7% |
| tree | 5/5 | 5/5 | 90.54% | 83.7-98.3% |
| umbrella | 5/5 | 5/5 | 92.14% | 83.3-97.8% |
| clock | 5/5 | 5/5 | 97.20% | 90.8-99.7% |

## Observations

- All 25 supplied examples have the intended category ranked first.
- Car drawings have the lowest average top-1 confidence: 58.90%. The fourth car is still correctly ranked first at 47.2%, ahead of cat at 30.2% and umbrella at 15.2%. A five-class prediction does not need a score above 50% to rank first.
- Clock drawings have the highest average top-1 confidence: 97.20%.
- Confidence and accuracy measure different things. The mean displayed top-1 confidence is 80.71%, while observed top-1 accuracy on this sample is 100%. This does not establish confidence calibration.

## Limitations

- Only 25 supplied drawings, with five per category, were evaluated.
- Several drawings have similar shapes and styles; the sample does not establish performance across different users or arbitrary drawings.
- Only the five supported categories were tested; out-of-category rejection was not evaluated.
- These results are separate from the previously reported 1500-image held-out Quick, Draw! test.

No broad claim of 100% real-world accuracy should be made from this small set. Larger tests with more varied drawings and different users would provide stronger evidence.

## Individual screenshot results

| Screenshot | Intended | First prediction | Confidence | Second prediction | Confidence | Third prediction | Confidence | Top-1 correct |
| --- | --- | --- | ---: | --- | ---: | --- | ---: | --- |
| `cat1.png` | cat | cat | 62.8% | tree | 25.4% | umbrella | 7.1% | Yes |
| `cat2.png` | cat | cat | 78.4% | tree | 14.4% | car | 4.1% | Yes |
| `cat3.png` | cat | cat | 62.2% | tree | 22.3% | umbrella | 8.1% | Yes |
| `cat4.png` | cat | cat | 55.8% | tree | 21.6% | car | 12.6% | Yes |
| `cat5.png` | cat | cat | 64.7% | car | 20.3% | tree | 11.8% | Yes |
| `car1.png` | car | car | 60.4% | cat | 18.3% | umbrella | 17.5% | Yes |
| `car2.png` | car | car | 49.8% | umbrella | 27.8% | cat | 14.8% | Yes |
| `car3.png` | car | car | 83.7% | umbrella | 6.2% | cat | 5.9% | Yes |
| `car4.png` | car | car | 47.2% | cat | 30.2% | umbrella | 15.2% | Yes |
| `car5.png` | car | car | 53.4% | cat | 21.1% | umbrella | 18.1% | Yes |
| `tree1.png` | tree | tree | 89.5% | cat | 6.5% | umbrella | 3.2% | Yes |
| `tree2.png` | tree | tree | 91.9% | umbrella | 5.9% | cat | 1.9% | Yes |
| `tree3.png` | tree | tree | 89.3% | cat | 5.5% | umbrella | 4.5% | Yes |
| `tree4.png` | tree | tree | 83.7% | umbrella | 10.4% | cat | 5.7% | Yes |
| `tree5.png` | tree | tree | 98.3% | umbrella | 1.1% | cat | 0.5% | Yes |
| `umbrella .png` | umbrella | umbrella | 97.8% | tree | 1.4% | clock | 0.6% | Yes |
| `umbrella 2.png` | umbrella | umbrella | 83.3% | tree | 11.8% | cat | 4.2% | Yes |
| `umbrella 3.png` | umbrella | umbrella | 86.6% | tree | 10.6% | cat | 2.3% | Yes |
| `umbrella 4.png` | umbrella | umbrella | 95.9% | tree | 3.4% | clock | 0.4% | Yes |
| `umbrella 5.png` | umbrella | umbrella | 97.1% | tree | 2.2% | clock | 0.4% | Yes |
| `Clock.png` | clock | clock | 99.7% | cat | 0.2% | umbrella | 0.1% | Yes |
| `Clock 2.png` | clock | clock | 99.2% | cat | 0.4% | umbrella | 0.4% | Yes |
| `Clock 3.png` | clock | clock | 97.0% | umbrella | 2.4% | cat | 0.6% | Yes |
| `Clock 4.png` | clock | clock | 90.8% | umbrella | 5.7% | cat | 3.1% | Yes |
| `Clock 5.png` | clock | clock | 99.3% | cat | 0.4% | umbrella | 0.3% | Yes |

## Short report paragraph

I tested the app with 25 new drawings, five for each category, and saved screenshots of the results. In the 25 supplied screenshots, the correct category was ranked first every time, giving 100% top-1 and top-3 accuracy on this small set. The model was less confident about cars than clocks. For example, one car was ranked first with a score of 47.2%. These results do not mean the app will be 100% accurate on all drawings because the sample was small and several drawings had similar styles. The separate held-out Quick, Draw! test achieved 95.13% top-1 accuracy and 99.47% top-3 accuracy.

## Suggested repository placement

Place this Markdown file and manual_drawing_results.json in the project's results folder. Preserve the 25 screenshot files in a results/manual_drawings folder, keeping the original filenames used in the JSON. The screenshot directory is not included in these two analysis files.
