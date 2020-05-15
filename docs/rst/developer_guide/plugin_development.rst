.. _sec-developer_guide-plugins:

Developing a plugin
===================

Eradiate implements several of its kernel components in the form of *plugins*. Plugins are autonomous library components which inherit from an interface defined in one of the kernel libraries and can be instantiated dynamically by Eradiate's plugin manager based on plugin parameters. Plugins do not have to be known to the Eradiate application at compile time: they are discovered at run time.

Plugin development requires usage of macros which we will present in this guide.

This guide demonstrates the development of a plugin in the Eradiate kernel, here named Mitsuba,
because of its origins. During the following sections, a C++ application is developed which
imports a newly created plugin, after that creation of a new plugin based on an existing interface
is detailed, followed by a section on the definition of new interfaces. Finally the guide describes
the core steps to create a new library for the Eradiate kernel.

Using plugins in a C++ application
----------------------------------

Let us imagine that we want to write an application which instantiates a plugin object and uses one of its interface's member functions. In this example, we will see how to use Mitsuba's plugin manager to carry out these simple tasks.

Writing the C++ code
^^^^^^^^^^^^^^^^^^^^

We create an application which will instantiate a ``Toy`` plugin (see :ref:`writing-a-plugin` and :ref:`defining-a-new-interface`), from the ``toy`` library (see :ref:`adding-a-new-library`). This application consists of a single source file ``src/apptoy/apptoy.cpp``. We start by including the headers we will need.

.. code-block:: cpp

    // Required for any Mitsuba app
    #include <mitsuba/mitsuba.h>
    // Basic core objects needed to have the plugin system work
    #include <mitsuba/core/logger.h>
    #include <mitsuba/core/thread.h>
    #include <mitsuba/core/plugin.h>
    #include <mitsuba/core/properties.h>
    // Toy plugin interface
    #include <mitsuba/toy/toy.h>

- ``logger.h`` and ``thread.h`` include the functionalities to create a threaded application and use Mitsuba's logging system;
- ``plugin.h`` grants access to the plugin manager, which manages the instantiation of plugin objects;
- ``properties.h`` provides the ``Properties`` class, which holds the parameters passed to the plugin manager and in turn to the classes that are instantiated.
- ``toy.h`` provides the ``Toy`` interface, from which the desired plugin is derived.


The ``MTS_NAMESPACE_BEGIN(mitsuba)`` statement is then used to reduce code verbosity.

.. code-block:: cpp
    
    MTS_NAMESPACE_BEGIN(mitsuba)

We then declare a templated function ``example_func`` which will perform the specific task we think about. In our case, it will instantiate an ``Example`` ``Toy`` plugin and call its ``print`` method. ``example_func`` is parametrised by the ``Float`` and ``Spectrum`` types, just many other Mitsuba classes. This means that the compiler will generate one version of our function per declared variant in the ``mitsuba.conf`` file.

.. code-block:: cpp

    template <typename Float, typename Spectrum>
    void example_func() {
        using Toy = Toy<Float, Spectrum>;
        ref<Toy> example = PluginManager::instance()->create_object<Toy>(Properties("Example"));
        example->print();
    }

The ``using Toy = Toy<Float, Spectrum>`` statement reduces verbosity by aliasing the otherwise long ``Toy<Float, Spectrum>`` type. The next line instantiates a ``Toy`` plugin based on a ``Properties`` instance, which contains the plugin name as its first argument. It calls the plugin manager's ``create_object`` template member function to this end. We note that the plugin manager is implemented as a singleton and that we access its single instance using the ``instance`` static function. The third line calls the ``print`` member function.

Our task is defined; however, it still is just a template. We need to instantiate it. This is done using the ``MTS_INVOKE_VARIANT`` macro. Its first argument is the relevant string's variant, the second is the function's identifier and the others are the arguments to be passed to the instantiated template function. We also must not forget to perform the static initialisation of the ``Thread`` and ``Logger`` classes.

.. code-block cpp

    int main() {
        Thread::static_initialization();
        Logger::static_initialization();
        MTS_INVOKE_VARIANT("scalar_rgb", example_func);
        return 0;
    }

``MTS_INVOKE_VARIANT`` selects the appropriate  ``Float`` and ``Spectrum`` type based on the variant's string (*i.e.* ``scalar_rgb``), instantiates the ``example_func`` template consistently and calls it.

Combining all the above parts:

.. code-block:: cpp

    #include <mitsuba/mitsuba.h>

    #include <mitsuba/core/logger.h>
    #include <mitsuba/core/thread.h>
    #include <mitsuba/core/plugin.h>
    #include <mitsuba/core/properties.h>

    #include <mitsuba/toy/toy.h>

    using namespace mitsuba;

    template <typename Float, typename Spectrum>
    void example_func() {
        using Toy = Toy<Float, Spectrum>;
        ref<Toy> example = PluginManager::instance()->create_object<Toy>(Properties("Example"));
        example->print();
    }

    int main() {
        Thread::static_initialization();
        Logger::static_initialization();
        MTS_INVOKE_VARIANT("scalar_rgb", example_func);
        return 0;
    }

Writing the build (CMake) code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first write a CMake script to build our application. This script will be located at ``src/apptoy/CMakeLists.txt`` We declare an executable target with the only source file we created.

.. code-block:: cmake

    add_executable(apptoy apptoy.cpp)

Then, we link our application to the required libraries. We need to link to the ``core`` (for the plugin, logging and threading facilities, and more) and ``toy`` (for the  ``Toy`` plugins) libraries.

.. code-block:: cmake

    target_link_libraries(apptoy PRIVATE mitsuba-core mitsuba-toy)

We then register our application to the distribution directory.

.. code-block:: cmake

    add_dist(apptoy)

Finally, we make sure that our app will search for plugins in its root directory as well as in the OS's library directories when it is built on MacOS.

.. code-block:: cmake

    if (APPLE)
        set_target_properties(apptoy PROPMTSIES INSTALL_RPATH "@executable_path")
    endif()

The complete application CMake script is as follows:

.. code-block:: cmake

    add_executable(apptoy apptoy.cpp)

    target_link_libraries(apptoy PRIVATE mitsuba-core mitsuba-toy)

    add_dist(apptoy)

    if (APPLE)
        set_target_properties(apptoy PROPMTSIES INSTALL_RPATH "@executable_path")
    endif()

We then simply have to register this application's directory for build in the ``src/CMakeLists.txt`` file:

.. code-block:: cmake

    # ...
    # Mitsuba executables
    # ...
    add_subdirectory(apptoy)
    # ...

And that's it!

.. _writing-a-plugin:

Writing a plugin
----------------

Writing a plugin for an existing interface requires the creation of a source file (``.cpp``) file. They are located in source subdirectories named after the corresponding interface. In this example, we consider an ``example`` plugin, implemented by an ``Example`` template, itself deriving from the ``Toy`` interface (see :ref:`defining-a-new-interface`).

Writing the C++ code
^^^^^^^^^^^^^^^^^^^^

The ``Toy`` interface is defined in a library called ``toy``. It has a single public pure virtual method ``print`` which is intended to display a message. Our plugin implementation will therefore derive from ``Toy`` and implement the ``print`` method. In addition to this pure virtual method, a ``toy`` plugin must implement a constructor taking a ``Properties`` map as an argument.

We create our plugin file ``example.cpp`` in the ``src/toys`` directory. Our source file must include three headers:

- ``mitsuba/toy/toy.h`` contains definitions for the ``Toy`` interface;
- ``mitsuba/core/properties.h`` contains definitions for the ``Properties`` class, required due to the fact that plugins must be constructible from a ``Properties`` object;
- ``iostream`` is the standard header for stream manipulation (we want to print stuff to the terminal with our ``print`` method).

.. code-block:: cpp

    #include <mitsuba/core/properties.h> // Required for constructor
    #include <mitsuba/toy/toy.h>         // Toy interface definitions
    #include <iostream>                   // Required to print to terminal

Our plugin is implemented by the ``Example`` class template, which is parametrised by the ``Float`` and ``Spectrum`` types. The C++ compiler will take care of the generation of the different variants of our plugin upon compilation based on the declared Mitsuba variants (see the ``mitsuba.conf`` file).

.. code-block:: cpp

    NAMESPACE_BEGIN(mitsuba) // Plugin code must be in the mitsuba namespace

    template<typename Float, typename Spectrum>
    class Example : public Toy<Float, Spectrum> { // Forward template parameters to parent class
    public:

We start by importing base class definitions using the ``MTS_IMPORT_BASE`` macro. This notably defines locally the ``Base`` type, which we will use later. The first argument of ``MTS_IMPORT_BASE`` is the parent class name, and the following arguments are the parent class's data member names. Since ``Toy`` doesn't have any data member, we only pass the first argument.

.. code-block:: cpp

    MTS_IMPORT_BASE(Toy) // Import base class definitions

Since most plugins usually use components from Mitsuba's other libraries, we then locally make explicitly visible the most useful types. For that purpose, the ``MTS_IMPORT_TOY_TYPES`` macro is used (see :ref:`adding-a-new-library` for further information about this macro).

.. code-block:: cpp

    MTS_IMPORT_TOY_TYPES() // Import useful library types

Then, we define a constructor from a ``Properties`` object.

.. code-block:: cpp

    Example(const Properties &props) : Base(props), m_store(1.0) { }

This constructor doesn't do much, apart from calling the base class's constructor and assigning a default value to the ``m_store`` data member (see below). We then implement the ``print`` method, which simply writes the value of the ``m_store`` data member to the standard output.

.. code-block:: cpp

    void print() override {
        std::cout << "Value of m_store: " << m_store << "\n";
    }

Note the ``override`` keyword which makes clear that this function implements a virtual method (pure virtual, in this case).

We then call the ``MTS_DECLARE_CLASS`` macro (this is required for all plugins because they derive from the ``Object`` class). We then use the ``MTS_IMPLEMENT_CLASS_VARIANT`` macro outside of the class definition scope to tell Mitsuba's RTTI (runtime type inspection) that ``Example`` implements the ``Toy`` interface.

.. code-block:: cpp

        MTS_DECLARE_CLASS()
    };

    MTS_IMPLEMENT_CLASS_VARIANT(Example, Toy)

The ``m_store`` data member we mentioned earlier is then declared, with ``private`` access specification:

.. code-block:: cpp

    private:
        Float m_float;

We finally export our new plugin and provide some information about it, and close the ``mitsuba`` namespace:

.. code-block:: cpp

    MTS_EXPORT_PLUGIN(Example, "A toy example plugin") // Plugin declaration and description text

    NAMESPACE_END(mitsuba)

And that's it! The full contents of our plugin file are as follows:

.. code-block:: cpp

    #include <mitsuba/core/properties.h>
    #include <mitsuba/toy/toy.h>
    #include <iostream>

    NAMESPACE_BEGIN(mitsuba)

    template<typename Float, typename Spectrum>
    class Example : public Toy<Float, Spectrum> {
    public:
        MTS_IMPORT_BASE(Toy)
        MTS_IMPORT_TOY_TYPES()

        Example(const Properties &props) : Base(props), m_float(1.0) { }

        void print() override {
            std::cout << "Value of m_float: " << m_float << "\n";
        }

        MTS_DECLARE_CLASS()

    private:
        Float m_float;
    };

    MTS_IMPLEMENT_CLASS_VARIANT(Example, Toy)
    MTS_EXPORT_PLUGIN(Example, "A toy example plugin")

    NAMESPACE_END(mitsuba)

Writing the build (CMake) code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The defined plugin is useless until it is built. Mitsuba provides convenience CMake functions to ease the writing of a ``CMakeLists.txt`` CMake script for our plugin:

.. code-block:: cpp

    set(MTS_PLUGIN_PREFIX "toys")

    add_plugin(example example.cpp)

The ``add_plugin`` function takes the plugin's target name as its first argument, and the associated header and source files as its other arguments. And that's all! CMake will make sure that our plugin is built when we compile Mitsuba.

**Note:** Make sure that the ``src/toys`` plugin directory is included in ``src/CMakeLists.txt`` (see :ref:`defining-a-new-interface`).

.. _defining-a-new-interface:

Defining a new interface
------------------------

The ``Toy`` interface for which we wrote a plugin in :ref:`writing-a-plugin` is defined in the ``toy`` library. We'll talk about the details of this library in :ref:`adding-a-new-library` and only focus on the definition of the ``Toy`` interface.

Writing the C++ code
^^^^^^^^^^^^^^^^^^^^

To define the ``Toy`` interface, we will write a header file and a source file. The header for the ``toy`` library are located in the ``include/mitsuba/toy`` directory.

We start with an include guard and header includes required to access both library common type declarations and the ``Object`` interface definitions.

.. code-block:: cpp

    #pragma once                      // Header guard
    #include <mitsuba/toy/fwd.h>     // Library forward declarations
    #include <mitsuba/core/object.h> // Object interface definitions

We then open the ``mitsuba`` namespace and declare the ``Toy`` interface as a class template inheriting from ``Object``. Templating delegates to the compiler the work of creating a template instance for each Mitsuba variant during the build process. In addition, we use the ``MTS_EXPORT_TOY`` macro which sets appropriate symbol visibility for our interface class.

.. code-block:: cpp

    NAMESPACE_BEGIN(mitsuba)

    template <typename Float, typename Spectrum>
    class MTS_EXPORT_TOY Toy : public Object {

We then import locally types useful in the ``toy`` library using the ``MTS_IMPORT_TOY_TYPES`` macro.

.. code-block:: cpp

    public:
        MTS_IMPORT_TOY_TYPES()

We then define a single pure virtual ``print`` method, which must be implemented by all plugins deriving from this interface.

.. code-block:: cpp

        virtual void print() = 0;

At this point, we use the ``MTS_DECLARE_CLASS`` to make Mitsuba's RTTI aware of the existence of the ``Toy`` class in the class hierarchy (this is required from any class deriving from ``Object``).

.. code-block:: cpp

        MTS_DECLARE_CLASS()

We then declare a constructor from a ``Properties`` object, as well as a virtual destructor.

.. code-block:: cpp

    protected:
        Toy(const Properties& props);
        virtual ~Toy() override;
    };

This ends the ``Toy`` interface definitions. We finally use the ``MTS_EXTERN_CLASS_TOY`` macro to declare that the ``Toy`` class template is to be imported and not instantiated. We finally close the ``mitsuba`` namespace.

.. code-block:: cpp

    MTS_EXTERN_CLASS_TOY(Toy)
    NAMESPACE_END(mitsuba)

The complete header file is as follows:

.. code-block:: cpp

    #pragma once

    #include <mitsuba/toy/fwd.h>
    #include <mitsuba/core/object.h>

    NAMESPACE_BEGIN(mitsuba)

    template <typename Float, typename Spectrum>
    class MTS_EXPORT_TOY Toy : public Object {
    public:
        MTS_IMPORT_TOY_TYPES()

        /// Print a message to the terminal
        virtual void print() = 0;

        MTS_DECLARE_CLASS()

    protected:
        /// Create a new Toy
        Toy(const Properties& props);

        /// Virtual destructor
        virtual ~Toy() override;

    protected:
        // Protected data members
    };

    MTS_EXTERN_CLASS_TOY(Toy)
    NAMESPACE_END(mitsuba)

The source file ``src/libtoy/toy.cpp``, which defines the implementation of ``Toy``'s member functions, is much briefer and starts by including the ``toy.h`` header we just described:

.. code-block:: cpp

    #include <mitsuba/toy/toy.h>

    NAMESPACE_BEGIN(mitsuba)

We then define the implementation of the constructor and destructor. We leave them to defaults. Note that these functions are templates, and declared as such thanks to the use of the ``MTS_VARIANT`` macro, which is a shorthand for ``template <typename Float, typename Spectrum>``:

.. code-block:: cpp

    MTS_VARIANT Toy<Float, Spectrum>::Toy(const Properties& props) {}
    MTS_VARIANT Toy<Float, Spectrum>::~Toy() {}

We then use the ``MTS_IMPLEMENT_CLASS_VARIANT`` macro to make the RTTI system aware that ``Toy`` inherits from ``Object``.

.. code-block:: cpp

    MTS_IMPLEMENT_CLASS_VARIANT(Toy, Object, "Toy")

The ``MTS_INSTANTIATE_CLASS`` macro then ensures than all variants of the template are instantiated correctly:

.. code-block:: cpp

    MTS_INSTANTIATE_CLASS(Toy)

And finally, we close the ``mitsuba`` namespace.

.. code-block:: cpp

    NAMESPACE_END(mitsuba)

The complete contents of our source file are then:

.. code-block:: cpp

    #include <mitsuba/toy/toy.h>

    NAMESPACE_BEGIN(mitsuba)

    MTS_VARIANT Toy<Float, Spectrum>::Toy(const Properties& props) {}
    MTS_VARIANT Toy<Float, Spectrum>::~Toy() {}

    MTS_IMPLEMENT_CLASS_VARIANT(Toy, Object, "Toy")
    MTS_INSTANTIATE_CLASS(Toy)

    NAMESPACE_END(mitsuba)

Writing the build (CMake) code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We simply need to add the files we just wrote to the ``toy`` library's CMake script:

.. code-block:: cpp

    add_library(mitsuba-toy-obj OBJECT
        ${INC_DIR}/fwd.h

        toy.cpp     ${INC_DIR}/toy.h
    )

.. _adding-a-new-library:

Adding a new library
--------------------

We assumed so far that the ``Toy`` interface belonged to an existing ``toy`` library. But what if we actually had to define it ourselves? This is what we'll see now.

Writing the C++ code
^^^^^^^^^^^^^^^^^^^^

We start by creating new directories:

- ``include/mitsuba/toy`` will host our C++ headers;
- ``src/libtoy`` will host our C++ source files.

We then write our library definitions. our ``toy`` library will define a single interface ``Toy`` (see :ref:`defining-a-new-interface`) in the ``include/toy/toy.h`` header file.

As mentioned in the interface tutorial, ``toy.h`` includes the ``toy`` library's forward declaration header ``include/toy/fwd.h``. This file contains a set of `forward declarations <https://en.wikipedia.org/wiki/Forward_declaration>`_, *i.e.* identifier declarations without definitions. They make the compiler aware of some identifier properties it needs to generate a binary file from our sources.

Forward declarations start with an include guard and the inclusion of the ``core`` library forward declaration header so that core types such as ``Float`` and ``Spectrum`` are available.

.. code-block:: cpp

    #pragma once
    #include <mitsuba/core/fwd.h>

We then open the ``mitsuba`` namespace and forward declare our interface class templates. Here, we add a single ``Toy`` interface (see :ref:`defining-a-new-interface` for a discussion of its implementation).

.. code-block:: cpp

    NAMESPACE_BEGIN(mitsuba)
    template <typename Float, typename Spectrum> class Toy;

Next, we define the ``MTS_IMPORT_TOY_TYPES`` macro, used in template classes to locally import types in templated classes. Our library being very simple, our import macro only imports core types. It would get more complex if we would have to use one of the ``toy`` library types outside of itself, *e.g.* to couple different plugins. We then close the ``mitsuba`` namespace.

.. code-block:: cpp

    #define MTS_IMPORT_TOY_TYPES() \
        MTS_IMPORT_CORE_TYPES()
    NAMESPACE_END(mitsuba)

The complete forward declaration file is therefore:

.. code-block:: cpp

    #pragma once

    #include <mitsuba/core/fwd.h>

    NAMESPACE_BEGIN(mitsuba)

    // Forward declare classes defined in this library
    template <typename Float, typename Spectrum> class Toy;

    // Define toy library types
    #define MTS_IMPORT_TOY_TYPES() \
        MTS_IMPORT_CORE_TYPES()

    NAMESPACE_END(mitsuba)

The rest of the C++ code is presented in :ref:`defining-a-new-interface`. Well, actually, not all of it: there are certain precautions we must take to make sure that our library will integrate nicely in the build system. First, we must make sure that our library's symbols will be exported the way they should. This is done by adding a module declaration macro to ``include/mitsuba/platform.h``. The ``MTS_BUILD_MODULE`` variable, used to select which export definitions are used, will be set in our CMake code.

.. code-block:: cpp

    #define MTS_MODULE_CORE   1
    #define MTS_MODULE_RENDER 2
    #define MTS_MODULE_UI     3
    #define MTS_MODULE_TOY    4

    // ...

    #if MTS_BUILD_MODULE == MTS_MODULE_TOY
    #  define MTS_EXPORT_TOY MTS_EXPORT
    #  define MTS_EXTERN_TOY extern
    #else
    #  define MTS_EXPORT_TOY MTS_IMPORT
    #  if defined(_MSC_VER)
    #    define MTS_EXTERN_TOY
    #  else
    #    define MTS_EXTERN_TOY extern
    #  endif
    #endif

The ``MTS_EXTERN_TOY`` macro is then used in the ``include/core/config.h`` file to define the ``MTS_EXTERN_CLASS_TOY`` and ``MTS_EXTERN_STRUCT_TOY`` macros, used to declare plugin interfaces (see :ref:`defining-a-new-interface`). The ``MTS_EXTERN_CLASS_TOY`` macro definitions are created during the configuration step, handled by the ``resources/scripts/configure.py`` script. We threrefore add to it the required code:

.. code-block:: python

    f.write('/// Declare that a "struct" template is to be imported and not instantiated\n')
        w('#define MTS_EXTERN_STRUCT_TOY(Name)')
        for index, (name, float_, spectrum) in enumerate(enabled):
            w('    MTS_EXTERN_TOY template struct MTS_EXPORT_TOY Name<%s, %s>;' % (float_, spectrum))
        f.write('\n\n')

        f.write('/// Declare that a "class" template is to be imported and not instantiated\n')
        w('#define MTS_EXTERN_CLASS_TOY(Name)')
        for index, (name, float_, spectrum) in enumerate(enabled):
            w('    MTS_EXTERN_TOY template class MTS_EXPORT_TOY Name<%s, %s>;' % (float_, spectrum))
        f.write('\n\n')

This really is all the code we needed to add to our C++ codebase (and what creates it). Now, let's move on to the CMake part.

Writing the build (CMake) code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We start by writing the code required to build our library. It sits in a new ``src/libtoy/CMakeLists.txt`` file. We start by declaring the corresponding include directory

.. code-block:: cmake

    set(INC_DIR "../../include/mitsuba/toy")

We then declare a new target which will build our library. We include all the files required to compile it, including headers. For the moment, we just have the forward declaration header.

.. code-block:: cpp

    add_library(mitsuba-toy-obj OBJECT
        ${INC_DIR}/fwd.h
    )

Naturally, and as specified in :ref:`defining-a-new-interface`, this target must have more files than that to build a library useful to anything.

We then build this object library into a shared library and define a series of properties required for correct integration into the rest of the system. Note that the ``MTS_BUILD_MODULE`` variable is set to ``MTS_MODULE_TOY`` (see previous section to see where it is used).

.. code-block:: cmake

    add_library(mitsuba-toy SHARED $<TARGET_OBJECTS:mitsuba-toy-obj>)
    set_property(TARGET mitsuba-toy-obj PROPMTSY POSITION_INDEPENDENT_CODE ON)
    set_target_properties(mitsuba-toy-obj mitsuba-toy PROPMTSIES FOLDER mitsuba-toy)
    target_compile_definitions(mitsuba-toy-obj PRIVATE -DMTS_BUILD_MODULE=MTS_MODULE_TOY)

We then link our shared library to its dependencies. Here, we link with Intel's thread building blocks and the ``core`` library.

.. code-block:: cmake

    target_link_libraries(mitsuba-toy PRIVATE tbb)
    target_link_libraries(mitsuba-toy PUBLIC mitsuba-core)

Finally, we ensure that our built target will be registered for copy to the ``dist`` directory.

.. code-block:: cmake

    add_dist(mitsuba-toy)

Our CMake build file finally looks like this (including sources for the ``Toy`` interface):

.. code-block:: cmake

    set(INC_DIR "../../include/mitsuba/toy")

    add_library(mitsuba-toy-obj OBJECT
        ${INC_DIR}/fwd.h

        toy.cpp     ${INC_DIR}/toy.h
    )

    add_library(mitsuba-toy SHARED $<TARGET_OBJECTS:mitsuba-toy-obj>)
    set_property(TARGET mitsuba-toy-obj PROPMTSY POSITION_INDEPENDENT_CODE ON)
    set_target_properties(mitsuba-toy-obj mitsuba-toy PROPMTSIES FOLDER mitsuba-toy)
    target_compile_definitions(mitsuba-toy-obj PRIVATE -DMTS_BUILD_MODULE=MTS_MODULE_TOY)

    # Link to Intel's Thread Building Blocks
    target_link_libraries(mitsuba-toy PRIVATE tbb)

    # Link to libcore
    target_link_libraries(mitsuba-toy PUBLIC mitsuba-core)

    # Copy to 'dist' directory
    add_dist(mitsuba-toy)

The last finishing touch we need is to add our library to the upper-level ``src/CMakeLists.txt``. We register our library, as well as the associated plugins (in advance, although there is no plugin in the directory at the moment).

.. code-block:: cmake

    # Mitsuba support libraries
    # ...
    add_subdirectory(libtoy)
    # ...

    # Plugins
    # ...
    add_subdirectory(toys)

Advanced: Handling dependencies between plugins
-----------------------------------------------

[**Coming soon**]

Plugin mantras
--------------

- All Mitsuba code must be scoped in the ``mitsuba`` namespace using the ``NAMESPACE_BEGIN(mitsuba)`` and ``NAMESPACE_END(mitsuba)`` macros.
- All plugin interfaces in Mitsuba derive from the ``Object`` class.
- Plugin class constructors from a ``Properties`` instance **must** be defined as ``public``. The class will otherwise not be considered instantiable and will not be added to the list of available plugin types.

Plugin macro notes
------------------

n release builds of Mitsuba, the default visibility of symbols is set to ``hidden``  to reduce the size of executables an libraries. However symbols that are linked to must be made visible to other parts of the code. Two macros are used to make symbols visible for linkage.

- ``MTS_EXPORT_TOY``: this macro sets a symbol's visibility to ``default``, when the build type (``MTS_BUILD_MODULE``) is set to ``MTS_MODULE_TOY``, which is the case for the plugin library.
- ``MTS_EXPORT_CORE``: this macro sets a symbol's visibility to ``default``, when the build type (``MTS_BUILD_MODULE``) is set to ``MTS_MODULE_CORE``, which is the case for the core library.