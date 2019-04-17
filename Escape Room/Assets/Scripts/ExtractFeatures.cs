using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using Accord.Math;
using Accord.Audio;

/// <summary>
/// OBSELETE: was used when using records from Audacity
/// </summary>
public class ExtractFeatures : MonoBehaviour {
    private List<double[]> _samples;
    private List<string> _classes;

    private int _frameLenght = 16000;

    private string[] _labels = new[] { "keys", "clap", "water" };
    private string rootFolder = @"D:\_Documents\#_Cours_TUT_2018_2019\Innovation Project\audio-recognition-game\sound_recognition\extracted_data\";
    // Start is called before the first frame update
    void Start() {

        // Delete all files in a directory    
        string[] files = Directory.GetFiles(rootFolder);
        foreach (string file in files) {
            File.Delete(file);
            Console.WriteLine($"{file} is deleted.");
        }


        _samples = new List<double[]>();
        _classes = new List<string>();
        MelFrequencyCepstrumCoefficient mfcc = new MelFrequencyCepstrumCoefficient(lowerFrequency: 20, 
            upperFrequency: 8000, windowLength: 0.04, frameRate: 50, numberOfBins: 2048);
        foreach (var label in _labels) {
            AudioClip c = Resources.Load<AudioClip>("data/" + label + "_16");
            Debug.Log(c);
            var counter = 0;
            var data = new float[c.samples];
            c.GetData(data, 0);
            for (int j = 0; j < c.samples / _frameLenght + 1; j++)
            {
                double[] d;
                if (c.samples >= (j + 1) * _frameLenght) {
                    d = new double[_frameLenght];
                    for (int k = 0; k < _frameLenght; k++) {
                        d[k] = data[k + (j * _frameLenght)];
                    }
                    _samples.Add(d);
                    _classes.Add(label);
                } else {
                    var diff = (j + 1) * _frameLenght - c.samples;
                    d = new double[_frameLenght];
                    var index = j * _frameLenght;
                    //Debug.Log(c.samples);
                    for (int k = index; k < c.samples; k++) {
                        //Debug.Log(k + " " + _frameLenght + " " + diff + " " + (k + (j * _frameLenght)) + " " + c.samples);
                        d[k - index] = data[k];
                    }

                    for (int k = c.samples - index; k < _frameLenght; k++) {
                        d[k] = 0;
                    }

                    _samples.Add(d);
                    _classes.Add(label);
                }

                if (d.Max() > 0.1) {
                    Signal audio = Signal.FromArray(d, 16000);
                    var spectrogram = mfcc.Transform(audio).ToArray();
                    string outputMatrix = "";

                    for (int i = 1; i < 13; i++) {
                        for (int k = 0; k < 50; k++)
                        {
                            outputMatrix += spectrogram[k].Descriptor[i].ToString().Replace(',', '.');
                            if (k != 49)
                            {
                                outputMatrix += ";";
                            }
                        }
                        outputMatrix += "\n";
                    }

                    File.WriteAllText(rootFolder + label + "_" + counter++ + "_features.csv", outputMatrix);
                }
            }
        }
        
    }

}
