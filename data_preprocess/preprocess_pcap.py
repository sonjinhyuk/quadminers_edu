import shutil

from data_preprocess.pre_utils.cicflowmeter.flow_session import generate_session_class
from pre_utils import create_directory
from scapy.sessions import DefaultSession
from scapy.utils import PcapReader, tcpdump
from scapy.interfaces import (
    resolve_iface,
)
from scapy.config import conf
from scapy.data import ETH_P_ALL

from pathlib import Path
from tqdm import tqdm
import os
import argparse
import multiprocessing


def analsis_socket(sockets, sniff_sockets, session, run_index, count=0):
    dead_sockets = []
    lfilter = None
    stop_filter = None
    continue_sniff = True
    close_pipe = None

    for s in sockets:
        run_index += 1
        if s is close_pipe:  # type: ignore
            break
        try:
            p = s.recv()
        except EOFError:
            # End of stream
            try:
                s.close()
            except Exception:
                pass
            dead_sockets.append(s)
            continue
        except Exception as ex:
            msg = " It was closed."
            try:
                # Make sure it's closed
                s.close()
            except Exception as ex2:
                msg = " close() failed with '%s'" % ex2
            dead_sockets.append(s)
            if conf.debug_dissector >= 2:
                raise
            continue
        if p is None:
            continue
        if lfilter and not lfilter(p):
            continue
        p.sniffed_on = sniff_sockets[s]
        # on_packet_received handles the prn/storage
        session.on_packet_received(p)
        # check
        if (stop_filter and stop_filter(p)) or \
                (0 < count <= session.count):
            continue_sniff = False
            break
    # Removed dead sockets
    for s in dead_sockets:
        del sniff_sockets[s]
        if len(sniff_sockets) == 1 and \
                close_pipe in sniff_sockets:  # type: ignore
            # Only the close_pipe left
            del sniff_sockets[close_pipe]  # type: ignore
    return continue_sniff, run_index, session

def run_param(param):
    in_file = param['infile']
    output_file = param['output']
    output_mode = 'flow'
    url_model = ""
    temp_file = param['temp_file']
    session = generate_session_class(output_mode, temp_file, url_model)
    L2socket = None
    iface = None
    if not isinstance(session, DefaultSession):
        session = session or DefaultSession
        session = session(prn=None, store=False, **{})
    else:
        return
    sniff_sockets = {}
    karg = {'filter': 'ip and (tcp or udp)'}
    flt = karg.get('filter')
    offline = [in_file]
    if isinstance(offline, list) and  all(isinstance(elt, str) for elt in offline):
        sniff_sockets.update((PcapReader(
            fname if flt is None else
            tcpdump(fname,
                    args=["-w", "-"],
                    flt=flt,
                    getfd=True,
                    quiet=True)
        ), fname) for fname in offline)

    if not sniff_sockets or iface is not None:
        _RL2 = lambda i: L2socket or resolve_iface(i).l2listen()  # type: Callable[[_GlobInterfaceType], Callable[..., SuperSocket]]  # noqa: E501
        iface = iface or conf.iface
        sniff_sockets[_RL2(iface)(type=ETH_P_ALL, iface=iface, **karg)] = iface
    _main_socket = next(iter(sniff_sockets))
    select_func = _main_socket.select
    nonblocking_socket = getattr(_main_socket, "nonblocking_socket", False)
    close_pipe = None  # type: Optional[ObjectPipe[None]]

    def stop_cb():
        # type: () -> None
        continue_sniff = False

    stop_cb = stop_cb
    try:
        continue_sniff = True
        # Start timeout
        remain = None
        run_index = 0
        while sniff_sockets and continue_sniff:
            # if run_index % 1000 == 0:
            #     print(f"{run_index}" + "\t" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            sockets = select_func(list(sniff_sockets.keys()), remain)
            continue_sniff, run_index, session = analsis_socket(sockets, sniff_sockets, session, run_index)
        if len(session.flows) > 0:
            session.garbage_collect(None)
        shutil.move(temp_file, output_file)
    except KeyboardInterrupt:
        os.remove(temp_file)
        pass
    except Exception as e:
        print(e)
    for s in sniff_sockets:
        s.close()
    try:
        close_pipe.close()
    except:
        pass
    # results = session.toPacketList()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcap_dir', default='../pcaps/', type=str)
    parser.add_argument('--output_dir', default="../data/post/", type=str)
    parser.add_argument('--pass_count', default=0, type=int)
    parser.add_argument('--mode', default='file', type=str)
    parser.add_argument('--temp_dir', default="/tmp/threatattack/", type=str)
    parser.add_argument('--input_file', default="HTTPS_practice.pcap", type=str)
    parser.add_argument('--multiprocess', action='store_true')
    args = parser.parse_args()
    pcap_dir = args.pcap_dir
    output_dir = args.output_dir
    temp_dir = args.temp_dir
    mode = args.mode
    out_dir = output_dir
    create_directory(out_dir)
    num_cores = multiprocessing.cpu_count()
    param_list = []
    input_interface = None
    output_mode = 'flow'
    url_model = ""

    if mode == 'file':
        input_file = args.input_file
        pcap_name = input_file.split(".")[0]
        infile = f"{pcap_dir}{input_file}"
        output = f"{out_dir}{pcap_name}.csv"
        temp_dict = {
            "infile": infile,
            "output_mode": output_mode,
            "output": output,
            "url_model": url_model,
            "temp_file": f"{temp_dir}{pcap_name}.csv"
        }
        param_list.append(temp_dict)
    else:
        paths = Path(pcap_dir).rglob("**/*.pcap")
        newp = []
        args = parser.parse_args()

        for path in tqdm(paths, leave=False):
            infile = str(path)
            pcap_name = path.stem
            output = f"{out_dir}{pcap_name}.csv"
            if os.path.exists(output):
                continue
            temp_dict = {
                "infile": infile,
                "output_mode": output_mode,
                "output": output,
                "url_model": url_model,
                "temp_file": f"{temp_dir}{pcap_name}.csv"
            }
            param_list.append(temp_dict)
    for param in tqdm(param_list):
        run_param(param)
    # parmap.map(run_param, param_list, pm_pbar=True, pm_processes=num_cores//2)
