import matplotlib.pyplot as plt

class SliceViewer:

    def __init__(self, subject):

        self.subject = subject
        self.vertex  = self.subject.shape[0]


    def calculate_ratio(self, subject=None, view='axial'):

        if subject is None:
            subject = self.subject
            vertex = self.vertex

        else:
            vertex = subject.shape[0]

        ratios = []
        for i in range(0, vertex):
    
            slce = self._select_view(view=view, i=i)
            r = int((slce == 0).sum()) / vertex**2
            ratios.append(r)

        return ratios


    def triple_view(self, i=None):

        if i is None:
            i = self.vertex // 2

        saggital = self.subject[i, :, :]
        axial    = self.subject[:, i, :]
        coronal  = self.subject[:, :, i]

        fig, axes = plt.subplots(1, 3)
        for i, s in enumerate([saggital, axial, coronal]):
            axes[i].imshow(s.T, cmap="gray", origin="lower")
        plt.suptitle(f"{i}th Slice view for EPI image")
        plt.show()
        plt.close()


    def single_view(self, i, view='axial'):

        slce = self._select_view(view=view, i=i)

        fig, axes = plt.subplots()
        axes.imshow(slce.T, cmap="gray", origin="lower")
        plt.suptitle(f"{i}th {view.capitalize()} view for EPI image")
        plt.show()
        plt.close()


    def ratio_view(self, view='axial', threshold=.85):

        for i in range(0, self.vertex):
            
            slce = self._select_view(view=view, i=i)
            r = int((slce == 0).sum()) / self.vertex ** 2
            if r < threshold:
                title = f"{view.capitalize()} View for EPI image {i}th slice, {r:.3f} ratio"
                self._show_slice(slce, title)


    def multi_view(self, view='axial', step=5, subject=None):

        if subject is None:
            subject = self.subject
            vertex = self.vertex

        else:
            vertex = subject.shape[0]

        for i in range(0, vertex):
            
            if i % step == 0:
                title = f"{view.capitalize()} View for EPI image {i}th slice"
                self._show_slice(self._select_view(view, i), title=title)


    def _show_slice(self, slce, title=None):

        fig, axes = plt.subplots()
        axes.imshow(slce.T, cmap="gray", origin="lower")
        if title: plt.suptitle(title)
        plt.show()
        plt.close()


    def _select_view(self, view='axial', i=None, subject=None):
        '''
        # X: Saggital View
        # Y: Axial View
        # Z: Coronal View
        '''

        if subject is None:
            subject = self.subject

        if i is None:
            i = self.vertex // 2

        if view == 'saggital' : return subject[i, :, :]
        elif view == 'axial'  : return subject[:, i, :]
        elif view == 'coronal': return subject[:, :, i]
        else: pass