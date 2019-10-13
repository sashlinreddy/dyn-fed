"""Data utilities
"""
import os
import shutil
import sys
import tarfile
import zipfile

import six
from six.moves.urllib.error import HTTPError, URLError
from tqdm import tqdm

def _extract_archive(file_path, path='.', archive_format='auto'):
    """Extracts an archive if it matches .tar, .tar.gz
    """
    if archive_format == 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, six.string_types):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile
    
    if is_match_fn(file_path):
        with open_fn(file_path) as archive:
            try:
                archive.extractall(path)
            except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                if os.path.exists(path):
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        shutil.rmtree(path)

def my_hook(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to

def download_file(url, filename):
    """
    Load file from url
    """
    if sys.version_info[0] == 2:
        from urllib import urlretrieve # pylint: disable=no-name-in-module
    else:
        from urllib.request import urlretrieve

    print("Downloading %s ... " % filename)
    with tqdm(unit='B',
              unit_scale=True,
              unit_divisor=1024,
              miniters=1,
              desc=filename) as t:
        report_hook = my_hook(t)
        urlretrieve(url, filename, reporthook=report_hook)
    print("done!")

def get_file(fname,
             origin,
             untar=False,
             extract=False,
             cache_dir=None,
             cache_subdir='datasets',
             archive_format='auto'):
    """Downloads a file if it is not already in the cache
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.ftml')
        
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.ftml')
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False

    if not os.path.exists(fpath):
        print("File not found locally. Need to download")
        download = True
    else:
        print(f"File already exists at {fpath}. No need to download")

    if download:
        print(f"Downloading data from {origin}")

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                download_file(origin, fpath)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format='tar')
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath
