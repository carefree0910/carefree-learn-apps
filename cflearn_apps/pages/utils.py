import os
import time
import urllib.error
import urllib.request

from stqdm import stqdm
from typing import Any
from typing import Optional
from functools import wraps


def retry(exception: Any, tries: int = 4, delay: int = 3, backoff: int = 2):
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            tries_, delay_ = tries, delay
            while tries_ > 1:
                try:
                    return f(*args, **kwargs)
                except exception as e:
                    print(f"{e}, Retrying in {delay_} seconds...")
                    time.sleep(delay_)
                    tries_ -= 1
                    delay_ *= backoff
            return f(*args, **kwargs)

        return f_retry

    return deco_retry


@retry((urllib.error.HTTPError, ConnectionResetError))
def download_with_progress(url: str, tgt_path: str) -> None:
    folder = os.path.dirname(tgt_path)
    with DownloadProgressBar(
        unit="B",
        unit_scale=True,
        miniters=1,
        desc=url.split("/")[-1],
    ) as t:
        urllib.request.urlretrieve(url, tgt_path, reporthook=t.update_to)


class DownloadProgressBar(stqdm):
    def update_to(self, b: int, bsize: int, total: int):
        self.total = total
        self.update(min(self.total, b * bsize - self.n))

    def st_display(self, n: Optional[int], total: Optional[int], **kwargs: Any) -> None:
        if n is not None and total is not None:
            n = min(n, total)
        super().st_display(n, total, **kwargs)
