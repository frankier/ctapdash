from ctapdash.middleware import get_current_request


def set_onionskin_eeg(eeg):
    request = get_current_request()
    request.state["onionskin_eeg"] = eeg


def perform_monkeypatch():
    from mne.viz._mpl_figure import MNEBrowseFigure as MNEBrowseFigureOrig

    class MyMNEBrowseFigure(MNEBrowseFigureOrig):
        def __init__(self, *args, **kwargs):
            import numpy as np
            super().__init__(*args, **kwargs)
            onionskin_kwargs = {
                **self.mne.trace_kwargs,
            }
            self.mne.onionskins = self.mne.ax_main.plot(
                np.full((1, self.mne.n_channels), np.nan), **onionskin_kwargs
            )

        def _update_data(self):
            from mne.io.base import BaseRaw
            import numpy as np
            super()._update_data()
            request = get_current_request()
            if not request:
                return
            if "onionskin_eeg" not in request.state:
                self.mne.onionskin_data = None
                return
            onionskin_eeg = request.state["onionskin_eeg"]
            start, stop = self._get_start_stop()
            if isinstance(onionskin_eeg, BaseRaw):
                if stop is None:
                    data = onionskin_eeg[:, start:]
                else:
                    data = onionskin_eeg[:, start:stop]
                data = data[0]
            else:
                ix_start = np.searchsorted(
                    self.mne.boundary_times, self.mne.t_start - self.mne.sampling_period
                )
                ix_stop = ix_start + self.mne.n_epochs
                item = slice(ix_start, ix_stop)
                print(type(onionskin_eeg))
                data = np.concatenate(
                    onionskin_eeg.get_data(item=item, copy=False), axis=-1
                )
            data = self._process_data(
                data, start, stop, picks=self.mne.picks
            )
            self.mne.onionskin_data = data

        def _draw_traces(self):
            from matplotlib.colors import to_rgba_array
            from matplotlib.patches import Rectangle
            import numpy as np
            super()._draw_traces()
            if self.mne.onionskin_data is None:
                return
            picks = self.mne.picks
            offset_ixs = (
                picks
                if self.mne.butterfly and self.mne.ch_selections is None
                else slice(None)
            )
            offsets = self.mne.trace_offsets[offset_ixs]

            ch_colors = to_rgba_array(self.mne.ch_colors)
            ch_colors[:, 3] *= 0.5

            decim = np.ones_like(picks)
            data_picks_mask = np.isin(picks, self.mne.picks_data)
            decim[data_picks_mask] = self.mne.decim
            # decim can vary by channel type, so compute different `times` vectors
            decim_times = {
                decim_value: self.mne.times[::decim_value] + self.mne.first_time
                for decim_value in set(decim)
            }

            ch_names = self.mne.ch_names[picks]
            time_range = (self.mne.times + self.mne.first_time)[[0, -1]]
            ylim = self.mne.ax_main.get_ylim()
            for ii, line in enumerate(self.mne.onionskins):
                this_name = ch_names[ii]
                this_offset = offsets[ii]
                this_times = decim_times[decim[ii]]
                this_data = this_offset - self.mne.onionskin_data[ii] * self.mne.scale_factor
                this_data = this_data[..., :: decim[ii]]
                clip = 0.2 if self.mne.butterfly else 0.5
                bottom = max(this_offset - clip, ylim[1])
                height = min(2 * clip, ylim[0] - bottom)
                rect = Rectangle(
                    xy=np.array([time_range[0], bottom]),
                    width=time_range[1] - time_range[0],
                    height=height,
                    transform=self.mne.ax_main.transData,
                )
                line.set_clip_path(rect)
                line.set_xdata(this_times)
                line.set_ydata(this_data)
                color = ch_colors[ii]
                line.set_color(color)
                line.set_zorder(self.mne.zorder["data"] - 1)

    import mne.viz._mpl_figure
    mne.viz._mpl_figure.MNEBrowseFigure = MyMNEBrowseFigure

perform_monkeypatch()
