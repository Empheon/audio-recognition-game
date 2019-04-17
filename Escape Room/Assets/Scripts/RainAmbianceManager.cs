using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RainAmbianceManager : MonoBehaviour
{
    private float RainInterval = 5;
    private float Volume = 0.05f;
    public AudioClip LightRain;
    public AudioClip MediumRain;
    public AudioClip WindClip;

    private float _rainCounter;
    private LoopingAudioSource _lightLoopingAudioSource;
    private LoopingAudioSource _mediumLoopingAudioSource;
    private LoopingAudioSource _windLoopingAudioSource;
    private LoopingAudioSource _currentLoopingAudioSource;


    // Start is called before the first frame update
    void Start()
    {
        _lightLoopingAudioSource = new LoopingAudioSource(this, LightRain);
        _mediumLoopingAudioSource = new LoopingAudioSource(this, MediumRain);
        _windLoopingAudioSource = new LoopingAudioSource(this, WindClip);
        _currentLoopingAudioSource = _lightLoopingAudioSource;
        _currentLoopingAudioSource.Play(Volume);
        _windLoopingAudioSource.Play(0.02f);
    }

    // Update is called once per frame
    void Update()
    {
        if (_rainCounter < RainInterval + Random.Range(-5, 5))
        {
            _rainCounter += Time.deltaTime;
        }
        else
        {
            _rainCounter = 0;
            LoopingAudioSource newSource;
            if (Random.Range(0f, 1f) > 0.5)
            {
                newSource = _mediumLoopingAudioSource;
            }
            else
            {
                newSource = _lightLoopingAudioSource;
            }

            if (newSource != _currentLoopingAudioSource)
            {
                _currentLoopingAudioSource.Stop();
                _currentLoopingAudioSource = newSource;
                _currentLoopingAudioSource.Play(Volume);
                if (newSource == _lightLoopingAudioSource)
                {
                    _windLoopingAudioSource.Play(0.01f);
                }
                else
                {
                    _windLoopingAudioSource.Play(0.03f);
                }
            }
        }
        _currentLoopingAudioSource.Update();
        _windLoopingAudioSource.Update();
    }
}

/// <summary>
/// Provides an easy wrapper to looping audio sources with nice transitions for volume when starting and stopping
/// Code from DIGITAL RUBY (JEFF JOHNSON)
/// </summary>
class LoopingAudioSource {
    public AudioSource AudioSource { get; private set; }
    public float TargetVolume { get; private set; }

    public LoopingAudioSource(MonoBehaviour script, AudioClip clip) {
        AudioSource = script.gameObject.AddComponent<AudioSource>();
        AudioSource.loop = true;
        AudioSource.clip = clip;
        AudioSource.playOnAwake = false;
        AudioSource.volume = 0.0f;
        AudioSource.Stop();
        TargetVolume = 1.0f;
    }

    public void Play(float targetVolume) {
        if (!AudioSource.isPlaying) {
            AudioSource.volume = 0.0f;
            AudioSource.Play();
        }
        TargetVolume = targetVolume;
    }

    public void Stop() {
        TargetVolume = 0.0f;
    }

    public void Update() {
        if (AudioSource.isPlaying && (AudioSource.volume = Mathf.Lerp(AudioSource.volume, TargetVolume, Time.deltaTime)) == 0.0f) {
            AudioSource.Stop();
        }
    }
}
