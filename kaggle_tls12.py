"""Official CLI, temporary TLS1.2 transport; certificate verification retained.

The default route's TLS1.3 handshakes intermittently fail. This wrapper changes
only its own process; no proxy settings, credentials, or package code are edited.
"""
import ssl
import requests

original_session_init = requests.Session.__init__
original_pool_init = requests.adapters.HTTPAdapter.init_poolmanager


def session_init(self, *args, **kwargs):
    original_session_init(self, *args, **kwargs)
    self.trust_env = False


def pool_init(self, *args, **kwargs):
    ctx = ssl.create_default_context()
    ctx.minimum_version = ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    kwargs['ssl_context'] = ctx
    original_pool_init(self, *args, **kwargs)


requests.Session.__init__ = session_init
requests.adapters.HTTPAdapter.init_poolmanager = pool_init

if __name__ == '__main__':
    from kaggle.cli import main
    main()
