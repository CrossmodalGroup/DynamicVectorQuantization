import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

class DualGrainSeperatePermuter(pl.LightningModule):
    """
    use fine position to represent all-grain position
    seperate the coarse and fine position
    """
    def __init__(self, 
                 coarse_hw = 16,
                 fine_hw = 32,

                 content_pad_code = 1024, 
                 content_eos_code = 1025,

                 coarse_position_pad_code = 256,
                 coarse_position_eos_code = 257,
                 fine_position_pad_code = 1024,
                 fine_position_eos_code = 1025,

                 fine_position_order = "region-first",
                 ) -> None:
        super().__init__()
        self.hw1 = coarse_hw
        self.hw2 = fine_hw // coarse_hw
        self.fine_hw = fine_hw 
        self.hw2_square = int(self.hw2 * self.hw2)

        self.content_pad_code = content_pad_code
        self.content_eos_code = content_eos_code
        self.coarse_position_pad_code = coarse_position_pad_code
        self.coarse_position_eos_code = coarse_position_eos_code
        self.fine_position_pad_code = fine_position_pad_code
        self.fine_position_eos_code = fine_position_eos_code

        self.content_eos_tensor = self.content_eos_code * torch.ones(1).long()
        self.coarse_position_eos_tensor = self.coarse_position_eos_code * torch.ones(1).long()
        self.fine_position_eos_tensor = self.fine_position_eos_code * torch.ones(1).long()

        self.fine_position_order = fine_position_order
        assert self.fine_position_order in ["row-first", "region-first"]
        self.position_sequence_coarse = torch.from_numpy(np.array([i for i in range(int(coarse_hw ** 2))])).long()
        self.position_sequence_fine = torch.from_numpy(np.array([i for i in range(int(fine_hw ** 2))])).long().view(fine_hw, fine_hw)
        if self.fine_position_order == "region-first":
            self.position_sequence_fine = rearrange(self.position_sequence_fine, "(h1 h2) (w1 w2) -> h1 w1 (h2 w2)", h1=self.hw1, h2=self.hw2, w1=self.hw1, w2=self.hw2)

    def forward(self, indices, grain_indices):
        # grain_indices: 0 for coarse-grained (1 code) and 1 for fine-grained (4 codes)
        batch_size = indices.size(0)
        device = indices.device

        original_indices = indices.clone()
        indices = rearrange(indices, "B (h1 h2) (w1 w2) -> B h1 w1 (h2 w2)", h1=self.hw1, h2=self.hw2, w1=self.hw1, w2=self.hw2)

        # coarse-grain sequence
        ## coarse-content sequence
        coarse_content = indices[:, :, :, 0]
        coarse_content_list = [torch.cat([coarse_content[i][(grain_indices[i] == 0)].to(device), self.content_eos_tensor.to(device)]) for i in range(batch_size)]
        coarse_content_tensor = pad_sequence(coarse_content_list, batch_first=True, padding_value=self.content_pad_code)

        ## coarse-position sequence
        # coarse_position_list = []
        # for i in range(batch_size):
        #     position_sequence_coarse_i = self.position_sequence_coarse[grain_indices[i].view(-1).cpu() == 0].to(device)
        #     self.coarse_position_eos_tensor_i = self.coarse_position_eos_tensor.to(device)
        #     coarse_position_i = torch.cat([position_sequence_coarse_i, self.coarse_position_eos_tensor_i])
        #     coarse_position_list.append(coarse_position_i)
        coarse_position_list = [torch.cat([self.position_sequence_coarse[grain_indices[i].view(-1).cpu() == 0].to(device), self.coarse_position_eos_tensor.to(device)]) for i in range(batch_size)]
        coarse_position_tensor = pad_sequence(coarse_position_list, batch_first=True, padding_value=self.coarse_position_pad_code)

        ## coarse-segment sequence, all 0
        coarse_segment_tensor = torch.zeros_like(coarse_content_tensor).to(device).long()

        # fine-grain sequence
        if self.fine_position_order == "region-first":
            ## fine-content sequence
            fine_content_list = [torch.cat([indices[i][grain_indices[i] == 1].to(device).view(-1), self.content_eos_tensor.to(device)]) for i in range(batch_size)]
            fine_content_tensor = pad_sequence(fine_content_list, batch_first=True, padding_value=self.content_pad_code)

            ## fine-position sequence
            fine_position_list = [torch.cat([self.position_sequence_fine[grain_indices[i] == 1].view(-1).to(device), self.fine_position_eos_tensor.to(device)]) for i in range(batch_size)]
            fine_position_tensor = pad_sequence(fine_position_list, batch_first=True, padding_value=self.fine_position_pad_code)
        elif self.fine_position_order == "row-first":
            ## fine-content sequence
            fine_grain_indices = grain_indices.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
            fine_content_list = [torch.cat([original_indices[i][fine_grain_indices[i] == 1].view(-1).to(device), self.content_eos_tensor.to(device)]) for i in range(batch_size)]
            fine_content_tensor = pad_sequence(fine_content_list, batch_first=True, padding_value=self.content_pad_code)

            ## fine-position sequence
            fine_position_list = [torch.cat([self.position_sequence_fine[fine_grain_indices[i].cpu() == 1].to(device), self.fine_position_eos_tensor.to(device)]) for i in range(batch_size)]
            fine_position_tensor = pad_sequence(fine_position_list, batch_first=True, padding_value=self.fine_position_pad_code)
        else:
            raise NotImplementedError("{} is not supported yet!".format(self.fine_position_order))
        
        ## fine-segment sequence, all 1
        fine_segment_tensor = torch.ones_like(fine_content_tensor).to(device).long()

        return_dict = {
            "coarse_content": coarse_content_tensor,
            "fine_content": fine_content_tensor,
            "coarse_position": coarse_position_tensor,
            "fine_position": fine_position_tensor,
            "coarse_segment": coarse_segment_tensor,
            "fine_segment": fine_segment_tensor,
        }
        return return_dict
    
    def forward_back(self, coarse_content, fine_content, coarse_position, fine_position):
        batch_size, coarse_length = coarse_content.size()
        device = coarse_content.device
        fine_length = fine_content.size(1)

        target_coarse_idx = torch.zeros(batch_size, int(self.hw1) ** 2).to(device).long()
        target_idx = torch.zeros(batch_size, int(self.fine_hw) ** 2).to(device).long()

        for i in range(batch_size):
            for current_position in range(coarse_length):
                if coarse_position[i, current_position] == self.coarse_position_eos_code:
                    target_idx[i] = target_coarse_idx[i].repeat_interleave(4, dim=-1)
                    target_idx[i] = rearrange(target_idx[i], "(h1 w1 h2 w2) -> (h1 h2 w1 w2)", h1=self.hw1, h2=self.hw2, w1=self.hw1, w2=self.hw2)
                    break
                else:
                    target_coarse_idx[i, coarse_position[i, current_position]] = coarse_content[i, current_position]
                
            for current_position in range(fine_length):
                if fine_position[i, current_position] == self.fine_position_eos_code:
                    break
                else:
                    target_idx[i, fine_position[i, current_position]] = fine_content[i, current_position]
        
        target_idx = rearrange(target_idx, "B (h1 h2 w1 w2) -> B (h1 h2) (w1 w2)", h1=self.hw1, h2=self.hw2, w1=self.hw1, w2=self.hw2)
        return target_idx
    


if __name__ == "__main__":
    test_code = 2
    fine_position_order = "region-first"
    if test_code == 1:
        # test code 1
        x1 = torch.randint(0, 1024, (2, 8, 8))  # .cuda()
        x2 = 4 * torch.ones_like(x1)  # .cuda()

        grain_indices = torch.randint(0, 2, (2, 4, 4))  # .cuda()
        grain_indices_repeat = grain_indices.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
        
        original_indices = x1 * grain_indices_repeat + x2 * (1 - grain_indices_repeat)
        print(original_indices)
        # print(grain_indices)

        permuter = DualGrainSeperatePermuter(
            coarse_hw = 4,
            fine_hw = 8,

            content_pad_code = 1024, 
            content_eos_code = 1025,

            coarse_position_pad_code = 256,
            coarse_position_eos_code = 257,
            fine_position_pad_code = 1024,
            fine_position_eos_code = 1025,

            fine_position_order = fine_position_order,
        )
        out = permuter(original_indices, grain_indices)
        
        coarse_content, fine_content, coarse_position, fine_position = out["coarse_content"], out["fine_content"], out["coarse_position"], out["fine_position"]
        print(coarse_content)
        print(coarse_position)
        print(fine_content)
        print(fine_position)

        target_fine = permuter.forward_back(coarse_content, fine_content, coarse_position, fine_position)
        print(target_fine)
        print(torch.all(target_fine == original_indices))
    elif test_code == 2:
        ## test code 2
        original_indices = torch.tensor([
            [[ 936,  129,  129,  845,  845,  129,  845,  845,  369,  612,  310, 368,  156,  156,  670,  126,  612,  126,  845,  129,  613,  612,  10,  934,   81,  381,  381,  924,  204,  381,  381,  204],
            [ 936,  750,  129,  129,  129,  845,  845,  845,  936,   78,  808,  55,  808,  808,  204,  706,  596,  136,  596,  623,  204,  652, 612,  772,   81,  652,  535,  381,  652,  381,  924,  652],
            [ 903,  903,  903,  129,  129,  903,  212,  361,  695,  204,   55, 179,  623,  706,  110,  110,  154,  154,   34,   34,   55,  204, 999,  612,  306,  306,  381,  381,  310,  310,  381,  381],
            [ 903,  903,  750,  129,  728,  361,  535,    8,  433,  808,  706, 706,  596,  577,  110,  110,  154,  154,   34,   34,  310,  723, 204,  845,  306,  306,  381,  381,  310,  310,  381,  381],
            [ 728,  728,  936,  612,  129,  369,  310,  575,   78,  996,  606, 606,   69,   69,  131,  131,   73,   73,  740,  740,  721,  721, 508,  441,  126,  156,  381,  381,   81,   81,   81,   81],
            [ 728,  728,  999,  204,  773,  129,  433,  786,  529,   69,  606, 606,   69,   69,  131,  131,   73,   73,  740,  740,  721,  721, 310,  706,  623,  156,  381,  381,   81,   81,   81,   81],
            [ 126,  306,   69,  703,  652,  936,  670,  212,  256,  529,  606, 934,  773,  773,  152,  152,  817,  817,  606,  606,  212,  212, 703,  706,    0,    0,  381,  381,    0,    0,    0,    0],
            [ 703,  652,  652,  486,  156,  936,  310,  606,  442,  529,  996, 934,  773,  773,  152,  152,  817,  817,  606,  606,  212,  212, 606,  706,    0,    0,  381,  381,    0,    0,    0,    0],
            [ 178,  652,  156,  508,  178,  750,  310,  808,  775,  529,  577, 577,  817,  817,  919,  919,  529,  529,  577,  577,  703,  703, 178,  381,  136,   81,  577,  156,  204,  204,  204,  204],
            [ 999,  652,  652,  310,  381,  359,    8,  774,   78,  817,  577, 577,  817,  817,  919,  919,  529,  529,  577,  577,  703,  703, 606,  999,  845,  535,  178,  156,  204,  204,  204,  204],
            [ 652,  652,  670,  652,  750,  612,  486,  774,  996,  817,  817, 817,  817,  817,  442,  442,  306,  306,  703,  703,  817,  817, 703,  129,  903,  903,  204,  204,   81,   81,  381,  381],
            [ 508,  204,   81,  998,  568,  306,  129,   69,  253,   78,  817, 817,  817,  817,  442,  442,  306,  306,  703,  703,  817,  817, 808,  508,  903,  903,  204,  204,   81,   81,  381,  381],
            [ 652,  808,  652,  596,  903,  715,  998,  442,  703,  703,  934, 204,  446,  446,  773,  773,   81,   81,  703,   69,  204,  306, 204,    8,  840,  840,   81,   81,   81,   81,  840,  840],
            [ 156,  310,  652,   81,  845,  450,  195,   69,  703,  703,  486, 832,  446,  446,  773,  773,   81,   81,  606,  396,  832,  723, 212,  740,  840,  840,   81,   81,   81,   81,  840,  840],
            [   0,    0,  652,  381,  577,  612, 1009,  212,  204,  204,  110, 110,  359,  359,  897,  897,  343,  343,  845,  845,  839,  839, 212,  703,    0,    0,  652,  652,   81,   81,    0,    0],
            [   0,    0,  808,  204,   81,  596,  786,  212,  817,  817,  110, 110,  359,  359,  897,  897,  343,  343,  845,  845,  839,  839, 817,  703,    0,    0,  652,  652,   81,   81,    0,    0],
            [ 156,   55,  156,  508,  136,  136,  786,  577,  817,  897,  354, 354,  152,  152,  934,  934,  306,  306,   81,   81,  775,  775, 369,  369,    9,    9,  306,  786,    0,    0,   81,   81],
            [  81,  508,  156,  310,  136,  136,  741,    8,  703,  442,  354, 354,  152,  152,  934,  934,  306,  306,   81,   81,  775,  775, 369,  369,    9,    9,  508,   81,    0,    0,   81,   81],
            [  81,   81,  126, 1009,  256,  741,  832,  428,  817,  817,  736, 736,   30,   30,  396,  396,  741,  741,  319,  319,  306,  306,  81,  596,  762,  762,  744,  178,  178,  204,  204,   81],
            [ 652,  129,  680,   30,  596,  152,  354,  310,  817,  817,  736, 736,   30,   30,  396,  396,  741,  741,  319,  319,  306,  306,  81,  999,  762,  762,  131,  998,  596,  156,   81,  204],
            [ 680,  256,  354,  396,  762,  354,   69,  428,  703,  212,  442, 442,  343,  343,  152,  152,  680,  680,  919,  919,  934,  934, 154,  154,  773,  529,  773,   69,  466,  466,  381,   81],
            [ 354,  354,  762,  703,  741,  152,  773,  832,  703,  577,  442, 442,  343,  343,  152,  152,  680,  680,  919,  919,  934,  934, 154,  154,  354,  396,  817,   69,  466,  466,   81,  535],
            [ 773,  354,  832,  817,  396,  762,   69,  110,  606,  606,  577,  81,   69,   69,  820,  820,    0,    0,  319,  319,  577,  178, 368,  343,  817,  354,  606,  817,  750,  156,   81,  840],
            [ 131,  508,  596,   30,  552,  715,   34,  343,  368,  486,  606, 934,   69,   69,  820,  820,    0,    0,  319,  319,  606,  253, 715,  817,  256,  212,  773,  529,  178,  959,   81,   81],
            [ 750,  934,  774,  897,  817,  354,  529,   34,  919,  680,  486, 695,  486,  606,  306,  306,   34,   34,  820,  820,  774,   30, 817,  256,  311,  703,  529,  256,  204,  999,  195,   81],
            [ 606,  934,   69,  715,  817,  774,  897,  773,  156,  205,  354, 680,  596,  486,  306,  306,   34,   34,  820,  820,  817,  256, 414,  369,  529,  715,  901,  858,  786,  606,  711,  820],
            [ 606,  306,  212,  715,  354,  606,   30,   69,  256,  773,  817, 897,  919,  762,  623,  361,  361,  343,  354,  354,  354,  840, 773,  715,  538,  347,   19,  205,   73,  577,  703,  750],
            [ 361,  773,  212,  568,  354,  256,  703,  343,  256,  256,  256, 205,  774,  773,   30,   30,  256,  817,  568,  786,  587,  786, 680,  396,  840,  775,  189,  744,  934,  703,  577,  606],
            [ 178,  817,  212,  568,  354,  817,  256,  762,  204,  680,  369, 306,  897,  256,   69,   69,  354,  354,  744,  744,  744,  744, 354,  354,   30,  817,  817,  680,  934,  343,  703,  606],
            [ 577,  212,  934,  998,  575,  817,   69,  913,   30,  786,   69, 354,  204,  703,  343,  256,  354,  354,  744,  744,  744,  744, 587,  587,   73,  786,   69,  204,  934,  817,  306,  577],
            [  81,  817,  306,  381,  347,  369,  773,  152,  256,  414,  587, 897,  817,  919,  414,  840,  529,  762,  901,  354,  354,  354, 131,  156,  369,  131,  913,  178,  577,  577,  817,  817],
            [ 934,  817,  934,  486,  575,   69,  786,  897,  808,  773,   69, 817,  256,  808,  529,  680,  306,  529,   69,  817,  354,  354,  19,   19,  189,  744,  369,  129,  606,  606,  606,  446]],
            
            [
            [ 936,  129,  129,  845,  845,  129,  845,  845,  369,  612,  310, 368,  156,  156,  670,  126,  612,  126,  845,  129,  613,  612,  10,  934,   81,  381,  381,  924,  204,  381,  381,  204],
            [ 936,  750,  129,  129,  129,  845,  845,  845,  936,   78,  808,  55,  808,  808,  204,  706,  596,  136,  596,  623,  204,  652, 612,  772,   81,  652,  535,  381,  652,  381,  924,  652],
            [ 129,  999,  903,  129,  129,  903,  808,  808,  695,  204,   55, 179,  623,  706,  703,  603,  706,  999,   34,   34,   55,  204, 999,  612,  306,  306,  381,  381,  310,  310,  204,   55],
            [ 936,  129,  750,  129,  728,  361,  808,  808,  433,  808,  706, 706,  596,  577,  934,  110,  535,  596,   34,   34,  310,  723, 204,  845,  306,  306,  381,  381,  310,  310,  204,  381],
            [ 936,  129,  936,  612,  129,  369,  310,  575,  204,  204,  606,   8,  204,   69,  131,  131,   73,   73,  740,  740,  721,  721, 154,  154,  126,  126,  381,  381,   81,   81,   81,   81],
            [ 612,  728,  999,  204,  773,  129,  433,  786,  204,  204,   78, 606,  606,  740,  131,  131,   73,   73,  740,  740,  721,  721, 154,  154,  126,  126,  381,  381,   81,   81,   81,   81],
            [ 126,  306,   69,  703,  652,  936,  670,  212,  256,  529,  606, 934,  773,  773,  152,  152,  817,  817,  606,  606,  212,  212, 703,  706,    0,    0,  381,  381,    0,    0,    0,    0],
            [ 703,  652,  652,  486,  156,  936,  310,  606,  442,  529,  996, 934,  773,  773,  152,  152,  817,  817,  606,  606,  212,  212, 606,  706,    0,    0,  381,  381,    0,    0,    0,    0],
            [ 178,  652,  156,  508,  178,  750,  310,  808,  775,  529,  577, 577,  817,  817,  919,  919,  529,  529,  577,  577,  703,  703, 178,  381,  136,   81,  577,  156,  204,  204,  204,  204],
            [ 999,  652,  652,  310,  381,  359,    8,  774,   78,  817,  577, 577,  817,  817,  919,  919,  529,  529,  577,  577,  703,  703, 606,  999,  845,  535,  178,  156,  204,  204,  204,  204],
            [ 652,  652,  670,  652,  750,  612,  486,  774,  996,  817,  817, 817,  817,  817,  442,  442,  306,  306,  703,  703,  817,  817, 703,  129,  903,  903,  204,  204,   81,   81,  381,  381],
            [ 508,  204,   81,  998,  568,  306,  129,   69,  253,   78,  817, 817,  817,  817,  442,  442,  306,  306,  703,  703,  817,  817, 808,  508,  903,  903,  204,  204,   81,   81,  381,  381],
            [ 652,  652,    0,    0,  903,  715,  998,  442,  808,  934,  934, 204,  446,  446,  773,  773,   81,   81,  703,   69,  204,  306, 381,  381,  840,  840,   81,   81,   81,   81,  840,  840],
            [ 652,  652,    0,    0,  845,  450,  195,   69,  703,  703,  486, 832,  446,  446,  773,  773,   81,   81,  606,  396,  832,  723, 381,  381,  840,  840,   81,   81,   81,   81,  840,  840],
            [ 652,  156,  652,  381,  577,  612, 1009,  212,  204,  204,  110, 110,  359,  359,  897,  897,  343,  343,  845,  845,  612,  623, 212,  703,    0,    0,  652,  652,   81,   81,    0,    0],
            [ 204,   81,  808,  204,   81,  596,  786,  212,  817,  817,  110, 110,  359,  359,  897,  897,  343,  343,  845,  845,  728,  152, 817,  703,    0,    0,  652,  652,   81,   81,    0,    0],
            [ 156,   55,  156,  508,  178,  845,  786,  577,  817,  897,  354, 354,  152,  152,  934,  934,  306,  306,   81,   81,  775,  775, 369,  369,    9,    9,  306,  786,    0,    0,  204,   81],
            [  81,  508,  156,  310,  998,  820,  741,    8,  703,  442,  354, 354,  152,  152,  934,  934,  306,  306,   81,   81,  775,  775, 369,  369,    9,    9,  508,   81,    0,    0,  924,  156],
            [  81,   81,  126, 1009,  256,  741,  832,  428,  817,  817,  736, 736,   30,   30,  396,  396,  741,  741,  319,  319,  306,  306,  81,  596,  762,  256,  596,  596,  156,  156,  204,   81],
            [ 652,  129,  680,   30,  596,  152,  354,  310,  817,  817,  736, 736,   30,   30,  396,  396,  741,  741,  319,  319,  306,  306,  81,  999,  762,   34,  596,  596,  156,  156,   81,  204],
            [ 680,  256,  354,  396,  152,  152,   69,  428,  703,  212,  442, 442,  343,  343,  152,  152,  680,  680,  919,  919,  934,  934, 154,  154,  762,  762,  773,   69,  466,  466,  381,   81],
            [ 354,  354,  762,  703,  152,  152,  773,  832,  703,  577,  442, 442,  343,  343,  152,  152,  680,  680,  919,  919,  934,  934, 154,  154,  762,  762,  817,   69,  466,  466,   81,  535],
            [ 773,  354,  832,  817,  396,  762,   69,  110,  606,  606,  577,  81,   69,   69,  820,  820,    0,    0,  319,  319,  577,  178, 368,  343,  354,  354,  606,  817,  750,  156,   81,  840],
            [ 131,  508,  596,   30,  552,  715,   34,  343,  368,  486,  606, 934,   69,   69,  820,  820,    0,    0,  319,  319,  606,  253, 715,  817,  354,  354,  773,  529,  178,  959,   81,   81],
            [ 750,  934,  774,  897,  817,  354,  529,   34,  919,  680,  486, 695,  486,  606,  306,  306,   34,   34,  820,  820,  774,   30, 817,  256,  311,  703,  529,  256,  204,  999,  195,   81],
            [ 606,  934,   69,  715,  817,  774,  897,  773,  156,  205,  354, 680,  596,  486,  306,  306,   34,   34,  820,  820,  817,  256, 414,  369,  529,  715,  901,  858,  786,  606,  711,  820],
            [ 606,  306,  212,  715,  354,  606,   30,   69,  256,  773,  817, 897,  919,  762,  623,  361,  361,  343,  354,  354,  354,  840, 773,  715,  538,  347,   19,  205,   73,  577,  703,  750],
            [ 361,  773,  212,  568,  354,  256,  703,  343,  256,  256,  256, 205,  774,  773,   30,   30,  256,  817,  568,  786,  587,  786, 680,  396,  840,  775,  189,  744,  934,  703,  577,  606],
            [ 178,  817,  212,  568,  354,  817,  256,  762,  204,  680,  369, 306,  897,  256,   69,   69,   30,  131,  744,  744,  744,  744, 354,  354,   30,  817,  817,  680,  934,  343,  703,  606],
            [ 577,  212,  934,  998,  575,  817,   69,  913,   30,  786,   69, 354,  204,  703,  343,  256,  596,  762,  744,  744,  744,  744, 587,  587,   73,  786,   69,  204,  934,  817,  306,  577],
            [ 934,  934,  306,  381,  347,  369,  773,  152,  256,  414,  587, 897,  817,  919,  414,  840,  529,  762,  901,  354,  354,  354, 131,  156,  369,  131,  623,  623,  577,  577,  817,  817],
            [ 934,  934,  934,  486,  575,   69,  786,  897,  808,  773,   69, 817,  256,  808,  529,  680,  306,  529,   69,  817,  354,  354,  19,   19,  189,  744,  623,  623,  606,  606,  606,  446]]
            ])
        
        grain_indices = torch.tensor([
            [[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]],

            [[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
            [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1]]
            
            ])

        permuter = DualGrainSeperatePermuter(
            coarse_hw = 16,
            fine_hw = 32,

            content_pad_code = 1024, 
            content_eos_code = 1025,

            coarse_position_pad_code = 256,
            coarse_position_eos_code = 257,
            fine_position_pad_code = 1024,
            fine_position_eos_code = 1025,

            fine_position_order = fine_position_order,
        )
        out = permuter(original_indices, grain_indices)

        coarse_content, fine_content, coarse_position, fine_position = out["coarse_content"], out["fine_content"], out["coarse_position"], out["fine_position"]

        target_fine = permuter.forward_back(coarse_content, fine_content, coarse_position, fine_position)
        print(target_fine)
        print(torch.all(target_fine == original_indices))