#-*- coding:utf-8 -*-
# We now have enough context about the python interpreter to begin examming Byterun.
# There are four kinds of objects in Byterun:
# A VirtualMachine class, which manages the highest-level structure, particularly 
# the call stack frames, and contains a mapping of instructions to operations.This
# is a more complex version of the Interpreter object above.
# A Frame class. Every Frame instance has one code object and manages a few other 
# necessary bits of state, particularly the global and local namespaces, a reference
# to the calling frame, and the last bytecode instruction executed.
# A Function class, which will be used in place of real Python functions. Recall that 
# calling a function creates a new frame in the interpreter. We implement Function so
# that we control the creation of new Frames.
# A Block class, which just wraps the three attributes of blocks. (The details of blocks
# aren't central to the Python interpreter, so we won't spend much time on them, but they're
# includes here so that Byterun can run real Python code.)
import collections
import operator
import dis
import sys
import types
import inspect

class VirtualMachineError(Exception):
    pass

Block = collections.namedtuple("Block", "type, handler, stack_height")

class VirtualMachine(object):
    """ 管理最高层的结构，特别是调用栈，同时管理指令到操作的映射，是最开始写的Interpreter类的高级版本 """
    def __init__(self):
        # 调用栈
        self.frames = []    # The call stack of frames.
        self.frame = None   # The current frame
        # frame 返回时的返回值
        self.return_value = None 
        self.last_exception = None
    
    def run_code(self, code, global_names = None, local_names = None):
        """ An entry point to execute code using the virtual machine. """
        """ 运行python程序的入口，程序编译后生成code_obj,这里code_obj在参数code中，
            run_code根据输入的code_obj新建一个frame并开始运行 """
        frame = self.make_frame(code, global_names = global_names, local_names = local_names)
        val = self.run_frame(frame)
        if self.frames:
            raise VirtualMachineError("Frames left over!")
        if self.frame and self.frame.stack:
            raise VirtualMachineError("Data left on stack! %r" % self.frame.stack)
        return val
    
    # Frame manipulation
    # 新建一个帧，code 为code_obj，callargs为函数调用时的参数
    def make_frame(self, code, callargs = {}, global_names = None, local_names = None):
        if global_names is not None:
            global_names = global_names
            if local_names is None:
                local_names = global_names
        elif self.frames:
            global_names = self.frame.global_names
            local_names = {}
        else:
            global_names = local_names = {
                '__builtins__':__builtins__,
                '__name__':'__main__',
                '__doc__':None,
                '__package__': None,
            }
        #将函数调用时的参数更新到局部变量空间中
        local_names.update(callargs)
        frame = Frame(code, global_names, local_names, self.frame)
        return frame
    
    # 调用栈压入frame
    def push_frame(self, frame):
        self.frames.append(frame)
        self.frame = frame
    
    # 调用栈弹出frame
    def pop_frame(self):
        self.frames.pop()
        if self.frames:
            self.frame = self.frames[-1]
        else:
            self.frame = None
    
    # Data stack manipulation
    # 数据栈操作
    def top(self):
        return self.frame.stack[-1]
    
    def pop(self):
        return self.frame.stack.pop()
    
    def push(self, *vals):
        self.frame.stack.extend(vals)
    
    def popn(self, n):
        """ Pop a number of values from the value stack.
            A list of 'n' values is returned, the deepest value first.
        """
        # 弹出多个值
        if n:
            ret = self.frame.stack[-n:]
            self.frame.stack[-n:] = []
            return ret 
        else:
            return []
    
    def parse_byte_and_args(self):
        f = self.frame
        opoffset = f.last_instruction
        # 取得要运行的指令
        byteCode = f.code_obj.co_code[opoffset]
        f.last_instruction += 1
        # 指令名称
        byte_name = dis.opname[byteCode]
        arg = None
        arguments = []
        # 指令码 < dis.HAVE_ARGUMENT 的都是无参数指令，其它的则是有参数指令
        if byteCode >= dis.HAVE_ARGUMENT:
            # index into the bytecode
            # 取得后两字节的参数
            print(f.code_obj.co_code)
            arg = f.code_obj.co_code[f.last_instruction:f.last_instruction + 2]
            f.last_instruction += 2 # advance the instruction pointer
            # 参数的第一个字节为参数实际低位，第二个字节为参数实际高位
            print(arg)
            print(arg[0],"    ",arg[1])
            arg_val = arg[0] + (arg[1] << 8)
            # 查找常量
            if byteCode in dis.hasconst:    # Look up a constant
                print(f.code_obj.co_consts)
                print("arg_val = ", arg_val)
                arg = f.code_obj.co_consts[arg_val]
            # 查找变量名
            elif byteCode in dis.hasname:   # Look up a name
                arg = f.code_obj.co_names[arg_val]
            # 查找局部变量名
            elif byteCode in dis.haslocal:  # Look up a local name
                arg = f.code_obj.co_varnames[arg_val]
            # 计算跳转位置
            elif byteCode in dis.hasjrel:   # Calculate a relative jump
                arg = f.last_instruction + arg_val
            else:
                arg = arg_val
            argument = [arg]
        else:
            argument = []
        
        return byte_name, argument
        
    def dispatch(self, byte_name, argument):
        """ Dispatch by bytename to the corresponding methods.
            Exceptions are caught and set on the virtual machine. """
        # When later unwinding the block stack
        # We need to keep track of why we are doing it.
        why = None
        try:
            # 通过指令名得到对应的方法参数
            bytecode_fn = getattr(self, 'byte_%s' % byte_name, None)
            if bytecode_fn is None:
                # 这里对一元操作、二元操作和其它操作做了区分
                if byte_name.startswith('UNARY_'):
                    self.unaryOperator(byte_name[6:])
                elif byte_name.startswith('BINARY_'):
                    self.binaryOperator(byte_name[7:])
                else:
                    raise VirtualMachineError(
                        "unsupported bytecode type: %s" % byte_name
                    )
            else:
                why = bytecode_fn(*argument)
        except:
            # deal with exceptions encountered while executing the op.
            # 存储运行指令时的异常信息
            self.last_exception = sys.exc_info()[:2] + (None,)
            why = 'exception'
        
        return why
    def run_frame(self, frame):
        """ Run a frame until it returns (somehow).
        Exceptions are raised, the return value is returned.
        """
        # 运行帧直至它返回
        self.push_frame(frame)
        while True:
            byte_name, arguments = self.parse_byte_and_args()
            why = self.dispatch(byte_name, arguments)

            # Deal with any block management we need to do
            while why and frame.block_stack:
                why = self.manage_block_stack(why)
            if why:
                break
        self.pop_frame()
        if why == 'exception':
            exc, val, tb = self.last_exception
            e = exc(val)
            e.__traceback__ = tb
            raise e
        return self.return_value
    
    # Block stack manipulation
    def push_block(self, b_type, handler = None):
        stack_height = len(self.frame.stack)
        self.frame.block_stack.append(Block(b_type, handler, stack_height))
    
    def pop_block(self):
        return self.frame.block_stack.pop()
    
    def unwind_block(self, block):
        """ Unwind the values on the data stack corresponding to a given block. """
        if block.type == 'except-handler':
            # The exception itself is on the stack as type, value, and traceback.
            offset = 3
        else:
            offset = 0
        while len(self.frame.stack) > block.stack_height + offset:
            self.pop()
        if block.type == 'except-handler':
            traceback, value, exctype = self.popn(3)
            self.last_exception = exctype, value, traceback

    def manage_block_stack(self, why):
        """ """
        # 管理一个frame的block栈，在循环、异常处理、返回这几个方面操作block栈与数据栈
        frame = self.frame
        block = frame.block_stack[-1]
        if block.type == 'loop' and why == 'continue':
            self.jump(self.return_value)
            why = None
            return why
        
        self.pop_block()
        self.unwind_block(block)

        if block.type == 'loop' and why == 'break':
            why = None
            self.jump(block.handler)
            return why
        
        if (block.type in ['setup-except', 'finally'] and why == 'exception'):
            self.push_block('except-handler')
            exctype, value, tb = self.last_exception
            self.push(tb, value, exctype)
            self.push(tb, value, exctype)
            why = None
            self.jump(block.handler)
            return why
        
        elif block.type == 'finally':
            if why in ('return', 'continue'):
                self.push(self.return_value)
            self.push(why)
            why = None
            self.jump(block.handler)
            return why
        return why

    ## Stack manipulation
    def byte_LOAD_CONST(self, const):
        self.push(const)
    def byte_POP_TOP(self):
        self.pop()
    
    ## Names
    def byte_LOAD_NAME(self, name):
        frame = self.frame
        if name in frame.f_locals:
            val = frame.f_locals[name]
        elif name in frame.func_globals:
            val = frame.func_globals[name]
        elif name in frame.f_builtins:
            val = frmae.f_builtins[name]
        else:
            raise NameError("name '%s' is not defined" % name)
        self.push(val)
    def byte_STORE_NAME(self, name):
        self.frme.f_locals[name] = self.pop()
    
    def byte_LOAD_FAST(self, name):
        if name in self.frame.f_locals:
            val = self.frame.f_locals[name]
        else:
            raise UnboundLocalError(
                "local variable '%s' referenced before assignment" % name
            )
        self.push(val)
    
    def byte_STORE_FAST(self, name):
        self.frame.f_locals[name] = self.pop()
    
    def byte_LOAD_GLOBAL(self, name):
        f = self.frame
        if name in f.f_globals:
            val = f.f_globals[name]
        elif name in f.f_builtins:
            val = f.f_builtins[name]
        else:
            raise NameError("global name '%s' is not defined" % name)
        self.push(val)
    
    ## Operators

    BINARY_OPERATORS = {
        'POWER':    pow,
        'MULTIPLY': operator.mul,
        'FLOOR_DIVIDE': operator.floordiv,
        'TRUE_DIVIDE':  operator.truediv,
        'MODULO':   operator.mod,
        'ADD':      operator.add,
        'SUBTRACT': operator.sub,
        'SUBSCR':   operator.getitem,
        'LSHIFT':   operator.lshift,
        'RSHIFT':   operator.rshift,
        'AND':      operator.and_,
        'XOR':      operator.xor,
        'OR':       operator.or_,
    }

    def binaryOperator(self, op):
        x, y = self.popn(2)
        self.push(self.BINARY_OPERATORS[op](x, y))

    COMPARE_OPERATORS = [
        operator.lt,
        operator.le,
        operator.eq,
        operator.ne,
        operator.gt,
        operator.ge,
        lambda x, y: x in y,
        lambda x, y: x not in y,
        lambda x, y: x is y,
        lambda x, y: x is not y,
        lambda x, y: issubclass(x, Exception) and issubclass(x, y),
    ]

    def byte_COMPARE_OP(self, opnum):
        x, y = self.popn(2)
        self.push(self.COMPARE_OPERATORS[opnum](x, y))

    ## Attributes and indexing

    def byte_LOAD_ATTR(self, attr):
        obj = self.pop()
        val = getattr(obj, attr)
        self.push(val)

    def byte_STORE_ATTR(self, name):
        val, obj = self.popn(2)
        setattr(obj, name, val)

    ## Building

    def byte_BUILD_LIST(self, count):
        elts = self.popn(count)
        self.push(elts)

    def byte_BUILD_MAP(self, size):
        self.push({})

    def byte_STORE_MAP(self):
        the_map, val, key = self.popn(3)
        the_map[key] = val
        self.push(the_map)

    def byte_LIST_APPEND(self, count):
        val = self.pop()
        the_list = self.frame.stack[-count] # peek
        the_list.append(val)

    ## Jumps

    def byte_JUMP_FORWARD(self, jump):
        self.jump(jump)

    def byte_JUMP_ABSOLUTE(self, jump):
        self.jump(jump)

    def byte_POP_JUMP_IF_TRUE(self, jump):
        val = self.pop()
        if val:
            self.jump(jump)

    def byte_POP_JUMP_IF_FALSE(self, jump):
        val = self.pop()
        if not val:
            self.jump(jump)
    
    def jump(self, jump):
        self.frame.last_instruction = jump


    ## Blocks

    def byte_SETUP_LOOP(self, dest):
        self.push_block('loop', dest)

    def byte_GET_ITER(self):
        self.push(iter(self.pop()))

    def byte_FOR_ITER(self, jump):
        iterobj = self.top()
        try:
            v = next(iterobj)
            self.push(v)
        except StopIteration:
            self.pop()
            self.jump(jump)

    def byte_BREAK_LOOP(self):
        return 'break'

    def byte_POP_BLOCK(self):
        self.pop_block()

    ## Functions

    def byte_MAKE_FUNCTION(self, argc):
        name = self.pop()
        code = self.pop()
        defaults = self.popn(argc)
        globs = self.frame.f_globals
        fn = Function(name, code, globs, defaults, None, self)
        self.push(fn)

    def byte_CALL_FUNCTION(self, arg):
        lenKw, lenPos = divmod(arg, 256) # KWargs not supported here
        posargs = self.popn(lenPos)

        func = self.pop()
        frame = self.frame
        retval = func(*posargs)
        self.push(retval)

    def byte_RETURN_VALUE(self):
        self.return_value = self.pop()
        return "return"
    
    ## Prints
    def byte_PRINT_ITEM(self):
        item = self.pop()
        sys.stdout.write(str(item))
    def byte_PRINT_NEWLINE(self):
        print("")

    
class Frame(object):
    """ 每一个Frame对象都维护一个code object引用，并管理一些必要的状态信息，比如全局与局部的命名空间，以及
        对调用它自身的帧的引用和最后执行的字节码 """
    # Frame对象包括一个code object，局部，全局，内置(builtin)的名字空间(namespace),对调用它的帧的引用，
    # 一个数据栈，一个block栈以及最后运行的指令的序号（在code_obj字节码中的位置）。由于python在处理不同模块
    # 时对名字空间的处理方式可能不同，在处置内置名字空间时需要做一些额外的工作。
    def __init__(self, code_obj, global_names, local_names, prev_frame):
        self.code_obj = code_obj
        self.global_names = global_names
        self.local_names = local_names
        self.prev_frame = prev_frame
        # 数据栈
        self.stack = []
        if prev_frame:
            self.builtin_names = prev_frame.builtin_names
        else:
            self.builtin_names = local_names['__builtins__']
            if hasattr(self.builtin_names, '__dict__'):
                self.builtin_names = self.builtin_names.__dict__
        self.f_lineno = code_obj.co_firstlineno
        self.f_lasti = 0
        if code_obj.co_cellvars:
            self.cells = {}
            if not prev_frame.cells:
                prev_frame.cells = {}
            for var in code_obj.co_cellvars:
                cell = Cell(self.local_names.get(var))
                prev_frame.cells[var] = self.cells[var] = cell
        else:
            self.cells = None
        if code_obj.co_freevars:
            if not self.cells:
                self.cells = {}
            for var in code_obj.co_freevars:
                assert self.cells is not None
                assert prev_frame.cells, "prev_frame.cells: %r" % (prev_frame.cells,)
                self.cells[var] = prev_frame.cells[var]
            
        # 最后运行的指令，初始为0
        self.last_instruction = 0
        # block 栈
        self.block_stack = []
        self.generator = None

# 每次调用一个函数其实就是调用了对象的__call__方法，每次调用都新创建了一个Frame对象
# 并开始运行它
class Function(object):
    """
    Create a realistic function object, defining the things the interpreter expects.
    """
    # __slots__会固定对象的属性，无法再动态增加新的属性，这可以节省内存空间
    __slots__ = [
        'func_code', 'func_name', 'func_defaults', 'func_globals',
        'func_locals', 'func_dict', 'func_closure',
        '__name__', '__dict__',
        '_vm', '_func',
    ]
    def __init__(self, name, code, globs, defaults, closure, vm):
        """ You don't need to follow this closely to understand the interpreter. """
        self._vm = vm
        # 这里的code即所调用函数的code_obj
        self.func_code = code
        # 函数名会存在code.co_name中
        self.func_name = self.__name__ = name or code.co_name
        # 函数参数的默认值，如func(a = 5, b = 3), 则func_defaults为（5，3）
        self.func_defaults = tuple(defaults)
        self.func_globals = globs
        self.func_locals = self._vm.frame.func_locals
        self.__dict__ = {}
        # 函数的闭包信息
        self.func_closure = closure
        self.__doc__ = code.co_consts[0] if code.co_consts else None

        # Sometimes, we need a real Python function. This is for that
        # 有时我们需要用到真实的python函数，下面的代码是为它准备的
        kw = {
            'argdefs': self.func_defaults,
        }
        # 为闭包创建cell对象
        if closure:
            kw['closure'] = tuple(make_cell(0) for _ in closure)
        self._func = types.FunctionType(code, globs, **kw)
    
    def __repr__(self):
        return '<Function %s at 0x%08x>' % (
            self.func_name, id(self)
        )
    
    def __get__(self, instance, owner):
        if instance is not None:
            return Method(instance, owner, self)
        else:
            return self
    
    
    def __call__(self, *args, **kwargs):
        """ when calling a Function, make a new frame and run it. """
        # 每当调用一次函数，都会创建一个新frame并运行
        # 通过inspect获得函数的参数
        callargs = inspect.getcallargs(self._func, *args, **kwargs)
        # Use callargs to provide a mapping of arguments: values to pass into the 
        # new frame.
        # 创建函数的帧
        frame = self._vm.make_frame(
            self.func_code, callargs, self.func_globals, {}
        )
        CO_GENERATOR = 32 # flag for "this code uses yield"
        if self.func_code.co_flags & CO_GENERATOR:
            gen = Generator(frame, self._vm)
            frame.generator = gen
            retval = gen
        else:
            retval = self._vm.run_frame(frame)
        return retval
    
def make_cell(value):
    """ Create a real Python closure and grab a cell. """
    # 创建一个真实的cell对象
    # Thanks to Alex Gaynor for help with this bit of twistiness.
    fn = (lambda x: lambda: x)(value)
    return fn.__closure__[0]

if __name__ == '__main__':
    code = """
def loop():
    x = 1
    while x < 5:
        if x == 3:
            break
        x = x + 1
        print(x)
    return x
loop()
    """
    # compile 能够将源代码编译成字节码
    code_obj = compile(code, "tmp", "exec")
    vm = VirtualMachine()
    print(code_obj)
    vm.run_code(code_obj)

