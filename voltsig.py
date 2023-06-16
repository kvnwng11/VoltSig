import numpy as np
from scipy.integrate import nquad


class VoltSig:
    def __init__(self, path, kernel, T, num_elements=-1):
        """
        Params:
            path: a 2-d array. path[0] corresponds to X1
            kernel: kernel function to integrate. takes in 2 parameters
            T: terminal time value used in the computation of the signature
            num_intervals: the number of intervals in the path. used for the step value of integration, where 
                            step = T/num_intervals   
        """
        def func(*args):
            """
            Description: Function for kernel integration.

            Params:
                args = [t_0, ... , t_n, a_0, ... , a_m, ... , T, num_args] where the inner-most integral is integrated with respect to x0,.
                        a_0, ..., a_m are static parameters that shouldn't be accessed in this function.
                        num_args = m+1
            """
            global kernel_permutation
            global K
            # print(args)

            t_i = [args[i] for i in range(len(kernel_permutation)-1, -1, -1)]
            first_args = [args[i] for i in kernel_permutation]
            result = 1
            for i in range(len(first_args)):
                first = first_args[i]
                second = t_i[i]
                result *= self.K(first, second)
            return result

        # preprocess path if needed
        path = np.array(path)
        if path.ndim == 1:
            path = [path]

        # ensure everything is a numpy array
        # path[0] is the path of the inner-most integral
        self.orig_path = np.array(path)
        self.path = np.array(path)

        self.result = 0
        self.func = func
        self.T = T
        if self.path.ndim == 1:
            self.num_intervals = int((self.path).shape[0]) - 1
        else:
            self.num_intervals = int((self.path[0]).shape[0]) - 1
        self.lower = 0
        self.upper = self.num_intervals
        self.step = T/self.num_intervals
        self.num_dims = self.orig_path.shape[0]
        self.K = kernel
        self.num_elt = num_elements
        self.sig = []

    def gen_limits(self, curr_args: list, level_idx, lower, upper, dX) -> None:
        """
        Description: Prepares limits for the level_idx integral.

        Params:
            curr_args: arguments to help with determining limits. curr_args = [..., lower, upper, # args]
            level_idx: integral whose limits need to be prepared. Starts at 0 counting from the right
            Lower, upper: limits on integral left of level_idx
            limit_type[0]: innermost integral ... limit_type[d-1]: second-outermost integral
                        0 --> scalar; 1 --> variable
            dX: running dX value
        """

        # base case --> compute integral
        if level_idx == -1:
            calc = self.calc_nested_var_limits(curr_args)
            self.result += calc * dX
            # print("dX: ", dX)
            # print("Total: ", calc * dX)
            # print()
            return

        # print(level_idx)

        # special case for 1 level volt sig
        # should only be executed once for every call of compute_single
        if len(self.limit_type) == 0:
            lower_limit = lower
            curr_args = [lower_limit, lower_limit+self.step]
            curr_dX = self.path[level_idx][upper] - self.path[level_idx][lower]
            self.gen_limits(curr_args, level_idx-1, lower, upper, dX*curr_dX)
        # finds ranges for up to the variable limit (only for signature level >= 1)
        else:
            self.limit_type[level_idx] = 0
            lower_limit = 0
            for i in range(0, lower):
                curr_args = [lower_limit, lower_limit+self.step] + curr_args
                lower_limit += self.step
                curr_dX = self.path[level_idx][i+1] - self.path[level_idx][i]
                self.gen_limits(curr_args, level_idx-1, i, i+1, dX*curr_dX)
                curr_args = curr_args[2:]

            # find ranges including variable limit
            curr_args = [lower_limit] + curr_args
            self.limit_type[level_idx] = 1

            curr_dX = self.path[level_idx][upper] - self.path[level_idx][lower]
            self.gen_limits(curr_args, level_idx-1, lower, upper, dX * curr_dX)

            curr_args = curr_args[1:]
            self.limit_type[level_idx] = 0

        return

    def calc_nested_var_limits(self, args) -> float:
        """
        Description: Helper function to integrate over a path segment.

        Params:
            Args = [..., lower, upper, # args]. Helps with calculating limits of integration
            limit_type[0]: innermost integral ... limit_type[d-1]: second-outermost integral
                        0 --> scalar; 1 --> variable
        """
        global limit_type
        limit_type = self.limit_type
        #print("Limit types: ", limit_type)

        def range_scalar(*args):
            num_args = args[-1]
            scalar_count = 0
            var_count = 0
            num_limits = len(args) - num_args

            global limit_type
            for i in range(len(limit_type)-1, len(limit_type)-1-num_limits, -1):
                if limit_type[i] == 1:
                    var_count += 1
                else:
                    scalar_count += 1

            # i=-3 is where args[..., i, lower, upper, num_args]
            idx = -4 - var_count - 2*scalar_count
            lower = args[idx]
            upper = args[(idx+1)]

            if lower == upper:
                print("Error with index.")

            return (lower, upper)

        def range_var(*args):
            num_args = args[-1]
            scalar_count = 0
            var_count = 0
            num_limits = len(args) - num_args

            global limit_type
            for i in range(len(limit_type)-1, len(limit_type)-1-num_limits, -1):
                if limit_type[i] == 1:
                    var_count += 1
                else:
                    scalar_count += 1

            idx = -4 - var_count - 2*scalar_count
            lower = args[idx]
            upper = args[0]

            if lower == upper:
                print("Error with index.")

            return (lower, upper)

        def range_outer(*args):
            upper = args[-3]
            lower = args[-4]
            return (lower, upper)

        num_args = len(args)+2
        args = list(args) + [self.T, num_args]
        ranges = np.append(np.array([[range_scalar] if i == 0 else [
                           range_var] for i in self.limit_type]), [range_outer])
        r, _ = nquad(self.func, ranges, args=args)
        return r

    def compute_single(self, level) -> float:
        """
        Description: Computes a single element of the Volterra Signature.

        Params:
            level: level of signature element
        """
        self.result = 0
        lower = idx = 0
        # print(type(self.path[0]))
        # print("Self.path[0] shape:", (self.path[0]).shape)
        # print("Self.path[0]: ", self.path[0])
        # print("Self.path: ", self.path)
        # print(type(self.path))
        # print((self.path).shape)
        upper = self.num_intervals

        level -= 1  # level counting starts at 0
        for i in range(lower, upper):
            # print("Compute subtree: ", i, i+1)
            dX = self.path[-1][i+1] - self.path[-1][i]
            self.gen_limits(curr_args=[idx, idx+self.step],
                            level_idx=level-1, lower=i, upper=i+1, dX=dX)
            idx += self.step
        return self.result

    def gen_kernels(self, level, gen_kernel_level, idx: list):
        """
        Description: Generates all combinations of kernel functions and compute the signature. There are level! of them.

        Params:
            level: level of volterra integral
            gen_kernel_level: counts integrals starting from left
            idx: holds the indexes for the kernel function
        """
        if self.num_elt != -1 and len(self.sig) == self.num_elt:
            return

        if len(idx) == level:
            #print("DEBUG: Generated kernel permutation")
            global kernel_permutation
            kernel_permutation = idx
            result = self.compute_single(level)
            #print("DEBUG: Finished calculation :)")
            self.level_output += [result]
            self.sig += [result]
            return

        for tau in range(gen_kernel_level):
            i = -2 if tau == 0 else level-tau
            self.gen_kernels(level, gen_kernel_level+1, idx+[i])

        return

    def gen_path_combo(self, level, curr_idx, combo: list) -> None:
        """
        Description: Generates combinations of paths.

        Params:
            level: level of signature. Element of range [1, inf]
            curr_idx: which index in combo to change
        """
        if len(self.sig) == self.num_elt:
            return

        # base case: calculate all kernel varities of volt sig element
        if curr_idx == level:
            # print(combo)
            print("Calculating VoltSig element...")
            self.path = [[0]] * level
            for i in range(len(combo)):
                self.path[i] = self.orig_path[combo[i]-1]
            self.limit_type = (level-1)*[0]
            self.gen_kernels(level, 1, idx=[])
            return

        for i in range(self.num_dims):
            combo[curr_idx] += 1
            self.gen_path_combo(level, curr_idx+1, combo)
        combo[curr_idx] = 0
        return

    def calc(self, level) -> None:
        """
        Description:
            Driver function to calculate volterra signature. output[i] is the ith Volterra Signature.

        Params:
            d: level of volterra signature
        """
        print("Begin calculation...")
        self.d = level
        for level in range(1, d+1):
            self.level_output = []
            self.level = level
            self.gen_path_combo(level, 0, combo=[0 for i in range(level)])

    def get_sig(self):
        """
        Description: If self.num_elt is a positive number, returns the first self.num_elt elements of the signature as a numpy array. If self.num_elt is larger
        than the length of the signature, this function returns the entire signature. Else, returns the entire computed signature
        """
        output = self.sig
        # if len(output) < self.num_elt:
        #     diff = self.num_elt - len(output)
        #     last = output[-1]
        #     output += [last] * diff
        return output
