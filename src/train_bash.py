from llamafactory.train.tuner import run_exp
import socket
import requests
import traceback


def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


def _send_to_robot(webhook: str, msg):
    data = {
        "msgtype": "text",
        "text": {
            "content": msg
        }
    }
    requests.post(webhook, json=data)


def _send_alert(env, msg: str, webhook: str = 'https://oapi.dingtalk.com/robot/send?access_token=1aa95f3a249ecd6b29fe56287640476da5f4a443b96a21f676d8c0e7012fdd06'):
    host_name = socket.gethostname()
    try:
        machine_ip = socket.gethostbyname(host_name)
        _send_to_robot(f"{webhook}", f"【LLama Factory tuning Error:{env}】, " + f"{msg}" + f" host_ip: {machine_ip}")
    except:
        _send_to_robot(f"{webhook}", f"【LLama Factory tuning Error:{env}】, " + f"{msg}" + f" host_name: {host_name}")


if __name__ == "__main__":
    webhook = 'https://oapi.dingtalk.com/robot/send?access_token=1aa95f3a249ecd6b29fe56287640476da5f4a443b96a21f676d8c0e7012fdd06'
    try:
        main()
    except Exception as e:
        if webhook:
            _send_alert('LlamaFactory training failed', e, webhook)  # add function to judege the process
            traceback.print_exc()
        else:
            exit()
