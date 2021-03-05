import itertools
import functools
import time
import sys
import threading


def benchmark(func):
    '''
    デコレートした関数の実行時間を計測する関数
    '''
    @functools.wraps(func)
    def Wrapper(*arg, **kw):
        stime = time.clock()
        ret = func(*arg, **kw)
        etime = time.clock()
        print('{0}: {1:,f}ms'.format(func.__name__, (etime - stime) * 1000))

        return ret

    return Wrapper


def loading_animation(process_name, animation_type='circle'):
    '''
    デコレートした関数にアニメーションをつける関数

    Arguments:
        process_name: デコレート対象の関数名
    '''
    def _loading_animation(func):
        done = False

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t = threading.Thread(target=animation, args=(process_name,))
            t.setDaemon(True)
            t.start()
            res = func(*args, **kwargs)
            nonlocal done
            done = True

            time.sleep(0.7)
            return res

        def animation(s):
            load_anime = {
                'circle': (['|', '/', '-', '\\'], 0.1),
                'dot': (['.  ', '.. ', '...', '   '], 0.5)
            }
            at = animation_type if animation_type in load_anime else 'circle'
            for c in itertools.cycle(load_anime[at][0]):
                if done:
                    break
                sys.stdout.write('\r{0}: processing '.format(s) + c)
                sys.stdout.flush()
                time.sleep(load_anime[at][1])
            sys.stdout.write('\r{0}: Done!                       \n'.format(s))

        return wrapper
    return _loading_animation
