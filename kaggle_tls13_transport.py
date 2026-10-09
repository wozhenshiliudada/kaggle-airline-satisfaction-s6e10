"""Official CLI with the existing credential-safe adapter, pinned to TLS 1.3.

Some current Kaggle endpoints fail this machine's TLS 1.2 handshake. This
process-local shim preserves verification and passes credentials over stdin.
It does not retry mutations or change account/proxy/library configuration.
"""
import kaggle_windows_transport as transport

original_run = transport.subprocess.run


def run(command, *args, **kwargs):
    if isinstance(command, list) and command and command[0] == 'curl.exe':
        command = list(command)
        if '--tlsv1.2' in command:
            command[command.index('--tlsv1.2')] = '--tlsv1.3'
        if '--tls-max' in command:
            command[command.index('--tls-max')+1] = '1.3'
    return original_run(command, *args, **kwargs)


transport.subprocess.run = run

if __name__ == '__main__':
    from kaggle.cli import main
    main()
