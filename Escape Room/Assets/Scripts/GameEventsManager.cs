using System;
using System.Collections;
using TMPro;
using UnityEngine;
using UnityEngine.Rendering.PostProcessing;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class GameEventsManager : MonoBehaviour
{
    public TextMeshProUGUI Timer;
    public GameObject LoosePlane;
    public GameObject WinPlane;
    public GameObject FinalDoor;

    public PostProcessProfile Profile;
    private int _exposureTarget = 0;
    private bool _triggerExposure = false;
    private AutoExposure _autoExposure;
    public Image EndPanel;
    private Color _targetColor;
    public GameObject WinText;
    public GameObject LooseText;

    private GameObject _endText;

    private float _time = 601f;

    public GameObject Character;
    public GameObject CinematicCanvas;
    private Animator _characterAnimator;
    public AudioClip Call1;
    private bool _inCinematic = true;

    public GameObject MusicAmbiance;
    private AudioSource _musicAudioSource;
    private AudioSource _myAudioSource;
 
    void Start()
    {
        _autoExposure = Profile.GetSetting<AutoExposure>();
        ActionController.SetBlocked(true);
        _characterAnimator = Character.GetComponent<Animator>();
        StartCoroutine(StartCinematic());
        _musicAudioSource = MusicAmbiance.GetComponent<AudioSource>();
        _musicAudioSource.volume = 0.02f;
        _myAudioSource = GetComponent<AudioSource>();
    }

    IEnumerator StartCinematic() {
        yield return new WaitForSeconds(2);
        _myAudioSource.clip = Call1;
        _myAudioSource.Play();
        yield return new WaitForSeconds(2.5f);
        _characterAnimator.SetTrigger("Cinematic");
    }

    void Update() {
        if (_endText && _endText.activeSelf && !Cursor.visible) {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        } else if (!_endText || !_endText.activeSelf) {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }

        if (_inCinematic && _characterAnimator.GetCurrentAnimatorStateInfo(0).IsName("character_waking_up") && _characterAnimator.GetCurrentAnimatorStateInfo(0).normalizedTime > 1) {
            _inCinematic = false;
            CinematicCanvas.SetActive(false);
            ActionController.SetBlocked(false);
            _characterAnimator.enabled = false;
            Timer.gameObject.SetActive(true);
            var newLuminance = new FloatParameter {
                value = 0
            };
            _autoExposure.maxLuminance = newLuminance;
            _autoExposure.minLuminance = newLuminance;
        }
        if (!_inCinematic) {
            if (!_myAudioSource.isPlaying && _musicAudioSource.volume <= 0.04999f) {
                _musicAudioSource.volume = Mathf.Lerp(_musicAudioSource.volume, 0.05f, Time.deltaTime);
            }


            _time -= Time.deltaTime;
            if (_time <= 0) {
                if (!_triggerExposure)
                    Loose();
            } else {
                var minutes = Mathf.Floor(_time / 60);
                var seconds = Mathf.Floor(_time - minutes * 60);
                string newTimerText = String.Concat((minutes < 10 ? "0" : ""), minutes, ":", (seconds < 10 ? "0" : ""), seconds);
                if (Timer.text != newTimerText) {
                    Timer.text = newTimerText;
                }
            }

            if (_triggerExposure && Math.Abs(_autoExposure.maxLuminance - _exposureTarget) > 0.00001) {
                var newLuminance = new FloatParameter {
                    value = _autoExposure.maxLuminance + _exposureTarget / 45f
                };
                _autoExposure.maxLuminance = newLuminance;
                _autoExposure.minLuminance = newLuminance;
                EndPanel.color = new Color(_targetColor.r, _targetColor.g, _targetColor.b,
                    Math.Abs(newLuminance.value) / 9f);
            } else if (_triggerExposure && _endText && !_endText.activeSelf) {
                _endText.SetActive(true);
            }
        }
    }

    public void Win()
    {
        StartCoroutine(FinalDoor.GetComponent<DoorRotationLite>().Move());
        ActionController.SetBlocked(true);
        WinPlane.SetActive(true);
        _targetColor = Color.white;
        _exposureTarget = -9;
        _triggerExposure = true;
        _endText = WinText;
    }

    public void Loose() {
        StartCoroutine(FinalDoor.GetComponent<DoorRotationLite>().Move());
        ActionController.SetBlocked(true);
        LoosePlane.SetActive(true);
        _targetColor = Color.black;
        _exposureTarget = 9;
        _triggerExposure = true;
        _endText = LooseText;
    }

    // Buttons actions
    public void LoadMainScene() {
        SceneManager.LoadScene("SampleScene");
    }

    public void LoadMenu() {
        SceneManager.LoadScene("Menu");
    }
}
