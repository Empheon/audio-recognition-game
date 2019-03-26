using System.Collections;
using System.Collections.Generic;
using System.Linq;
using TensorFlow;
using UnityEngine;
using UnityEngine.Assertions.Comparers;

public class AudioSamplesClassificationTest : MonoBehaviour
{
    private List<float[]> _samples;
    private List<string> _classes;

    public TextAsset GraphModel;
    private int _frameLenght = 22050;

    private string[] _labels = new[] {"keys", "clap"};
    // Start is called before the first frame update
    void Start()
    {
        _samples = new List<float[]>();
        _classes = new List<string>();
        foreach (var label in _labels)
        {
            for (int i = 1; i < 30; i++) {
                AudioClip c = Resources.Load<AudioClip>(label + "/" + label + "_" + i);
                Debug.Log(c);
                
                var data = new float[c.samples];
                c.GetData(data, 0);

                for (int j = 0; j < c.samples / _frameLenght + 1; j++)
                {
                    if (c.samples >= (j + 1) * _frameLenght)
                    {
                        var d = new float[_frameLenght];
                        for (int k = 0; k < _frameLenght; k++)
                        {
                            d[k] = data[k + (j * _frameLenght)];
                        }
                        _samples.Add(d);
                        _classes.Add(label);
                    }
                    else
                    {
                        var diff = (j + 1) * _frameLenght - c.samples;
                        var d = new float[_frameLenght];
                        var index = j * _frameLenght;
                        //Debug.Log(c.samples);
                        for (int k = index; k < c.samples; k++) {
                            //Debug.Log(k + " " + _frameLenght + " " + diff + " " + (k + (j * _frameLenght)) + " " + c.samples);
                            d[k - index] = data[k];
                        }

                        for (int k = c.samples - index; k < _frameLenght; k++)
                        {
                            d[k] = 0;
                        }

                        _samples.Add(d);
                        _classes.Add(label);
                    }
                }

            }
        }

        int m = 0;
        foreach (var sample in _samples)
        {
            using (var graph = new TFGraph()) {

                graph.Import(GraphModel.bytes);

                // We declare and run a session with our graph.
                var session = new TFSession(graph);
                var runner = session.GetRunner();

                //var data = new float[_frameLenght];
                //_audioSource.GetOutputData(data, 0);
                var tensorData = new float[1, sample.Length, 1];
                var avg = 0f;


                //MelSpectrogram gram = new MelSpectrogram();



                var max = sample.Max();
                var min = sample.Min();
                //float data;
                for (int i = 0; i < sample.Length; i++) {
                    tensorData[0, i, 0] = (sample[i] - min) / (max - min);
                    avg += sample[i];
                }



                avg /= sample.Length;
                //Debug.Log(avg);
                if (true) {
                    TFTensor input = tensorData;
                    runner.AddInput(graph["input_node_input"][0], input);
                    runner.Fetch(graph["output_node/Sigmoid"][0]);

                    var recurrentTensor = runner.Run()[0].GetValue() as float[,];

                    // We dispose of resources so our graph doesn't break down over
                    // time. IMPORTANT if you will repeatedly call the graph.
                    session.Dispose();
                    graph.Dispose();

                    int c = 0;
                    List<float> outputs = new List<float>();
                    foreach (var v in recurrentTensor)
                    {
                        outputs.Add(v);
                    }
                    string e = "";
                    if (outputs[0] > outputs[1] && outputs[0] > 0.1) {
                        e = _labels[1];
                    } else if (outputs[1] > outputs[0] && outputs[1] > 0.1) {
                        e = _labels[0];
                    }
                    Debug.Log(m + "- Estimated: " + e + " | Orig: " + _classes[m] + " | " + outputs[0] + " " + outputs[1]);
                    m++;
                }
            }
        }
    }
    
}
