import torch
import torch.nn as nn

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class PositionAwareSOSProvider(AbstractEncoder):
    # for unconditional training with dynamic grainularity quantized transformer
    def __init__(self, coarse_sos, coarse_pos_sos, fine_sos=None, fine_pos_sos=None, coarse_seg_sos=None, fine_seg_sos=None):
        super().__init__()
        self.coarse_sos = coarse_sos 
        self.fine_sos = fine_sos
        self.coarse_pos_sos = coarse_pos_sos
        self.fine_pos_sos = fine_pos_sos
        self.activate_seg = True if coarse_seg_sos is not None else False
        if self.activate_seg:
            self.coarse_seg_sos = coarse_seg_sos
            self.fine_seg_sos = fine_seg_sos

    def encode(self, x):
        # get batch size from data and replicate sos_token
        batch_size = x.size(0)
        device = x.device

        c_coarse = (torch.ones(batch_size, 1) * self.coarse_sos).long().to(device)
        if self.fine_sos is not None:
            c_fine = (torch.ones(batch_size, 1) * self.fine_sos).long().to(device)
        else:
            c_fine = None
        
        c_pos_coarse = (torch.ones(batch_size, 1) * self.coarse_pos_sos).long().to(device)
        if self.fine_pos_sos is not None:
            c_pos_fine = (torch.ones(batch_size, 1) * self.fine_pos_sos).long().to(device)
        else:
            c_pos_fine = None

        if self.activate_seg:
            c_seg_coarse = (torch.ones(batch_size, 1) * self.coarse_seg_sos).long().to(device)
            c_seg_fine = (torch.ones(batch_size, 1) * self.fine_seg_sos).long().to(device)
            return c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine
        
        return c_coarse, c_fine, c_pos_coarse, c_pos_fine, None, None

class ClassForContentOnlyPositionAwareSOSProvider(AbstractEncoder):
    # for class-conditional training with dynamic grainularity quantized transformer
    # compared with unconditional, we replace [coarse_sos, fine_sos] by class-label
    # class-label += threshold
    def __init__(self, n_classes, threshold, coarse_pos_sos, fine_pos_sos=None, coarse_seg_sos=None, fine_seg_sos=None):
        super().__init__()
        self.n_classes = n_classes
        self.threshold = threshold

        self.coarse_pos_sos = coarse_pos_sos
        self.fine_pos_sos = fine_pos_sos
        self.activate_seg = True if coarse_seg_sos is not None else False
        if self.activate_seg:
            self.coarse_seg_sos = coarse_seg_sos
            self.fine_seg_sos = fine_seg_sos

    def encode(self, x):
        # get batch size from data and replicate sos_token
        batch_size = x.size(0)
        device = x.device

        x = x[:,None]

        c_coarse = x + self.threshold
        if self.fine_pos_sos is not None:
            c_fine = x + self.threshold
        else:
            c_fine = None
        
        c_pos_coarse = (torch.ones(batch_size, 1) * self.coarse_pos_sos).long().to(device)
        if self.fine_pos_sos is not None:
            c_pos_fine = (torch.ones(batch_size, 1) * self.fine_pos_sos).long().to(device)
        else:
            c_pos_fine = None

        if self.activate_seg:
            c_seg_coarse = (torch.ones(batch_size, 1) * self.coarse_seg_sos).long().to(device)
            c_seg_fine = (torch.ones(batch_size, 1) * self.fine_seg_sos).long().to(device)
            return c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine
        
        return c_coarse, c_fine, c_pos_coarse, c_pos_fine, None, None

class ClassAwareSOSProvider(AbstractEncoder):
    # for class-conditional training with dynamic grainularity quantized transformer
    # compared with unconditional, we replace [coarse_sos, fine_sos, coarse_pos_sos, fine_pos_sos] by class-label
    # class-label += threshold
    def __init__(self, n_classes, threshold_content, threshold_coarse_position, threshold_fine_position, coarse_seg_sos=None, fine_seg_sos=None):
        super().__init__()
        self.n_classes = n_classes
        self.threshold_content = threshold_content
        self.threshold_coarse_position = threshold_coarse_position
        self.threshold_fine_position = threshold_fine_position

        self.activate_seg = True if coarse_seg_sos is not None else False
        self.coarse_seg_sos = coarse_seg_sos
        self.fine_seg_sos = fine_seg_sos

    def encode(self, x):
        # get batch size from data and replicate sos_token
        batch_size = x.size(0)
        device = x.device

        x = x[:,None]

        c_coarse = x + self.threshold_content
        if self.fine_seg_sos is not None:
            c_fine = x + self.threshold_content
        else:
            c_fine = None
        
        c_pos_coarse = x + self.threshold_coarse_position
        if self.fine_seg_sos is not None:
            c_pos_fine = x + self.threshold_fine_position
        else:
            c_pos_fine = None

        if self.activate_seg:
            c_seg_coarse = (torch.ones(batch_size, 1) * self.coarse_seg_sos).long().to(device)
            c_seg_fine = (torch.ones(batch_size, 1) * self.fine_seg_sos).long().to(device)
            return c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine
        
        return c_coarse, c_fine, c_pos_coarse, c_pos_fine, None, None