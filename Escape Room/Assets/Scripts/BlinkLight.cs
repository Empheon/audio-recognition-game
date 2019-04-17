using System.Collections;
using UnityEngine;

public class BlinkLight : MonoBehaviour
{
    private Light _light;

    private AudioSource _audioSource;

    private Renderer _renderer;
    // Start is called before the first frame update
    void Start()
    {
        _light = GetComponent<Light>();
        _renderer = transform.parent.GetComponent<Renderer>();
        _audioSource = GetComponent<AudioSource>();
        StartCoroutine(Blink());
    }

    public IEnumerator Blink() {
        float rnd = Random.Range(0.2f, 2f);
        yield return new WaitForSeconds(rnd);
        _light.enabled = !_light.enabled;
        _renderer.material.SetColor("_EmissionColor", _renderer.material.GetColor("_EmissionColor") == Color.black ? Color.white : Color.black);
        _audioSource.pitch = Random.Range(0.95f, 1.05f);
        _audioSource.Play();
        StartCoroutine(Blink());
    }
}
