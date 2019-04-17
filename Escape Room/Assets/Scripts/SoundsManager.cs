using UnityEngine;

public class SoundsManager : MonoBehaviour
{
    public AudioClip RipPaper;
    public AudioClip BookFall;
    public AudioClip FrameFall;
    private static AudioSource _audioSource;

    void Awake()
    {
        _audioSource = GetComponent<AudioSource>();
    }

    public void PlayRipPaper()
    {
        transform.position = new Vector3(-10.38f, 3f, -0.5f);
        _audioSource.clip = RipPaper;
        _audioSource.Play();
    }

    public void PlayBookFall()
    {
        transform.position = new Vector3(16.2f, 0.67f, -7.44f);
        _audioSource.clip = BookFall;
        _audioSource.PlayDelayed(0.8f);
    }

    public void PlayFrameFall()
    {
        transform.position = new Vector3(-10.38f, 0.68f, 0.37f);
        _audioSource.clip = FrameFall;
        _audioSource.PlayDelayed(0.8f);
    }
}
