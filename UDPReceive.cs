using UnityEngine;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;

public class UDPReceive : MonoBehaviour
{

    Thread receiveThread;
    UdpClient client; 
    public 
    public int port = 5052;
    public bool startRecieving = true;
    public bool printToConsole = false;
    public string data;
    private Animator animator;

    public string expressionType = "neutral";


    public void Start()
    {
        animator = GetComponent<Animator>();

        receiveThread = new Thread(
            new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }


    // receive thread
    private void ReceiveData()
    {

        client = new UdpClient(port);
        while (startRecieving)
        {

            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] dataByte = client.Receive(ref anyIP);
                expressionType = Encoding.UTF8.GetString(dataByte);

                if (printToConsole) { print(data); }
                switch (expressionType)
                {
                    case "happy":
                        // "happy" 표정을 나타내는 Blend Tree 파라미터 값을 설정합니다.
                        animator.SetFloat("exp", 1);
                        break;

                    case "sad":
                        // "sad" 표정을 나타내는 Blend Tree 파라미터 값을 설정합니다.
                        animator.SetFloat("exp", -1);
                        break;

                    default:
                        // 기본적으로 중립 표정을 나타내는 Blend Tree 파라미터 값을 설정합니다.
                        animator.SetFloat("exp", 0);
                        break;
                }
            }
            catch (Exception err)
            {
                print(err.ToString());
            }
        }
    }

}
