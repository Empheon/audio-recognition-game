using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class MenuActions : MonoBehaviour
{
    public GameObject LoadMenuObject;
    public Slider SliderObject;

    private AsyncOperation _async;
    private bool _startLoading;
    private AudioSource _audioSource;

    void Start() {
        _async = SceneManager.LoadSceneAsync("SampleScene");
        _async.allowSceneActivation = false;
        _audioSource = GetComponent<AudioSource>();
    }

    public void LoadGame() {
        StartCoroutine(Load());
        _startLoading = true;
    }

    private void Update() {
        // Loads too fast, fake loading bar to display controls
        if (_startLoading && SliderObject.value < 1) {
            Cursor.visible = false;
            SliderObject.value += Time.deltaTime / 10;
            _audioSource.volume = Mathf.Lerp(_audioSource.volume, 0, Time.deltaTime);
        }
    }

    IEnumerator Load() {
        LoadMenuObject.SetActive(true);
        yield return new WaitForSeconds(10);
        while (!_async.isDone) {
            //SliderObject.value = _async.progress;
            if (_async.progress >= 0.899f) {
                //SliderObject.value = 1;
                _async.allowSceneActivation = true;
            }
            yield return null;
        }
    }

    public void Exit() {
        Application.Quit();
    }
}
