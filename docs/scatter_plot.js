
const scatterPlotMain = () => {

    const data_location = './visualization_data/accuracy_vs_number_of_parameters.json';

    const svg = d3.select('#scatter-plot-svg');
    const scatterPlotGroup = svg.append('g');
    const attentionScatterPoints = scatterPlotGroup.append('g');
    const plainRNNScatterPoints = scatterPlotGroup.append('g');
    const scatterPlotTitle = scatterPlotGroup.append('text');
    const xAxisGroup = scatterPlotGroup.append('g');
    const xAxisLabel = xAxisGroup.append('text');
    const yAxisGroup = scatterPlotGroup.append('g');
    const yAxisLabel = yAxisGroup.append('text');
    const legend = scatterPlotGroup.append('g');
    const legendBoundingBox = legend.append('rect');
    const attentionLegendText = legend.append('text');
    const plainRNNLegendText = legend.append('text');
    const attentionLegendCircle = legend.append('circle');
    const plainRNNLegendCircle = legend.append('circle');
    const toolTipGroup = svg.append('g');
    const toolTipBoundingBox = toolTipGroup
          .append('rect')
          .style("opacity", 0);
    
    const getDatumLoss = datum => datum.test_loss;
    const getDatumParameterCount = datum => datum.number_of_parameters;

    const margin = {
        top: 80,
        bottom: 80,
        left: 220,
        right: 30,
    };

    const innerLineOpacity = 0.1;
    const xAxisRightPaddingAmount = 1000000;

    const scatterPointRadius = 3;
    const scatterPointFillOpacity = 0.3;
    const attentionFill = 'red';
    const attentionToolTipFill = '#ff99ad';
    const plainRNNFill = 'blue';
    const plainRNNToolTipFill = '#9cb3ff';
    const attentionLegendKeyText = 'Attention';
    const plainRNNLegendKeyText = 'Plain RNN';

    const toolTipTransitionTime = 250;
    const toolTipTextPadding = 10;
    
    const render = (attention_data, plain_rnn_data) => {
        
        const plotContainer = document.getElementById('scatter-plot');
        svg
            .attr('width', plotContainer.clientWidth)
            .attr('height', plotContainer.clientHeight);
        
        const svgWidth = parseFloat(svg.attr('width'));
        const svgHeight = parseFloat(svg.attr('height'));
        
        const innerWidth = svgWidth - margin.left - margin.right;
        const innerHeight = svgHeight - margin.top - margin.bottom;
        
        const xMaxValue = Math.max(d3.max(attention_data, getDatumParameterCount), d3.max(plain_rnn_data, getDatumParameterCount));
        const xScale = d3.scaleLinear()
              .domain([0, xMaxValue+xAxisRightPaddingAmount])
              .range([0, innerWidth]);
        
        const yScale = d3.scaleLinear()
              .domain([1.0, 0.0])
              .range([0, innerHeight]);
        
        scatterPlotGroup.attr('transform', `translate(${margin.left}, ${margin.top})`);
        
        const labelSize = Math.min(20, innerWidth/40);
        scatterPlotTitle
            .style('font-size', labelSize)
            .text('Test Loss vs Model Parameter Count')
            .attr('x', innerWidth * 0.325)
            .attr('y', -10);
        
        const legendKeyFontSize = Math.min(15, innerWidth/40);
        const attentionLegendKeyX = innerWidth - attentionLegendKeyText.length * legendKeyFontSize;
        const attentionLegendKeyY = innerHeight - legendKeyFontSize * 4.5;
        const plainRNNLegendKeyX = innerWidth - plainRNNLegendKeyText.length * legendKeyFontSize;
        const plainRNNLegendKeyY = innerHeight - legendKeyFontSize * 3;
        const legendBoundingBoxX = attentionLegendKeyX - legendKeyFontSize / 2;
        const legendBoundingBoxY = attentionLegendKeyY - legendKeyFontSize * 1.5;
        const legendBoundingBoxWidth = Math.max(attentionLegendKeyText.length, plainRNNLegendKeyText.length) * legendKeyFontSize * 0.75;
        const legendBoundingBoxHeight = legendKeyFontSize * 4;
        legendBoundingBox
            .attr('x', legendBoundingBoxX)
            .attr('y', legendBoundingBoxY)
            .attr('width', legendBoundingBoxWidth)
            .attr('height', legendBoundingBoxHeight)
            .style('stroke-width', 1)
            .style('stroke', 'black')
            .attr('fill', 'white');
        attentionLegendCircle
            .attr('cx', attentionLegendKeyX + legendKeyFontSize / 2)
            .attr('cy', attentionLegendKeyY - legendKeyFontSize * 0.75 + legendKeyFontSize / 2)
            .attr('r', legendKeyFontSize / 2)
            .attr('fill', attentionFill);
        plainRNNLegendCircle
            .attr('cx', plainRNNLegendKeyX + legendKeyFontSize / 2)
            .attr('cy', plainRNNLegendKeyY - legendKeyFontSize * 0.75 + legendKeyFontSize / 2)
            .attr('r', legendKeyFontSize / 2)
            .attr('fill', plainRNNFill);
        attentionLegendText
            .style('font-size', legendKeyFontSize)
            .html(attentionLegendKeyText)
            .attr('x', attentionLegendKeyX + legendKeyFontSize * 1.25)
            .attr('y', attentionLegendKeyY)
            .attr('stroke', attentionFill);
        plainRNNLegendText
            .style('font-size', legendKeyFontSize)
            .html(plainRNNLegendKeyText)
            .attr('x', plainRNNLegendKeyX + legendKeyFontSize * 1.25)
            .attr('y', plainRNNLegendKeyY)
            .attr('stroke', plainRNNFill);
        
        const yAxisTickFormat = number => d3.format('.3f')(number);
        yAxisGroup.call(d3.axisLeft(yScale).tickFormat(yAxisTickFormat).tickSize(-innerWidth));
        yAxisGroup.selectAll('.tick line')
            .style('opacity', innerLineOpacity);
        yAxisGroup.selectAll('.tick text')
            .attr('transform', 'translate(-3.0, 0.0)');
        yAxisLabel
            .style('font-size', labelSize * 0.8)
            .attr('fill', 'black')
            .attr('transform', 'rotate(-90)')
            .attr('y', -60)
            .attr('x', -innerHeight/3)
            .text('Mean Cross Entropy Loss');
        
        const xAxisTickFormat = number => d3.format('.3s')(number).replace(/G/,'B');
        xAxisGroup.call(d3.axisBottom(xScale).tickFormat(xAxisTickFormat).tickSize(-innerHeight))
            .attr('transform', `translate(0, ${innerHeight})`);
        xAxisGroup.selectAll('.tick line')
            .style('opacity', innerLineOpacity);
        xAxisGroup.selectAll('.tick text')
            .attr('transform', 'translate(0.0, 5.0)');
        xAxisLabel
            .style('font-size', labelSize * 0.8)
            .attr('fill', 'black')
            .attr('y', margin.bottom * 0.75)
            .attr('x', innerWidth / 2)
            .text('Parameter Count');

        const updateToolTip = (x, y, desiredOpacity, datum, backgroundColor) => {
            toolTipGroup.selectAll('g').remove();
            const toolTipTextLines = [
                `Number of Epochs: ${datum.number_of_epochs}`,
                `Batch Size: ${datum.batch_size}`,
                `Vocab Size: ${datum.vocab_size}`,
                `Pretrained Embedding Specification: ${datum.pre_trained_embedding_specification}`,
                `Encoding Hidden Size: ${datum.encoding_hidden_size}`,
                `Number of Encoding Layers: ${datum.number_of_encoding_layers}`,
            ];
            if (datum.attention_intermediate_size) {
                toolTipTextLines.push(
                    `Attention Intermediate Size: ${datum.attention_intermediate_size}`,
                    `Number of Attention Heads: ${datum.number_of_attention_heads}`
                );
            }
            toolTipTextLines.push(
                `Dropout Probability: ${datum.dropout_probability}`,
                `Test Loss: ${datum.test_loss}`,
                `Test Accuracy: ${datum.test_accuracy}`,
                `Number of Parameters: ${datum.number_of_parameters}`
            );
            const ephemeralTextLinesGroup = toolTipGroup.append('g');
            toolTipTextLines.forEach((textLine, textLineIndex) => {
                ephemeralTextLinesGroup
                    .append('text')
                    .style('font-size', labelSize)
                    .attr('class', 'displayed-text')
                    .attr('dx', toolTipTextPadding)
                    .attr('dy', `${(1+textLineIndex) * 1.2 * labelSize}px`)
                    .html(textLine);
            });
            const ephemeralTextLinesGroupBBox = ephemeralTextLinesGroup.node().getBBox();
            const toolTipBoundingBoxWidth = ephemeralTextLinesGroupBBox.width + 2 * toolTipTextPadding;
            const toolTipBoundingBoxHeight = ephemeralTextLinesGroupBBox.height + labelSize;
            const toolTipX = x < 0 ? x : (x + toolTipBoundingBoxWidth > svgWidth ? svgWidth - toolTipBoundingBoxWidth : x);
            const toolTipY = y;
            toolTipBoundingBox
                .attr('x', toolTipX)
                .attr('y', toolTipY)
                .style('stroke-width', 1)
                .style('stroke', 'black')
                .style('fill', backgroundColor)
                .attr('width', toolTipBoundingBoxWidth)
                .attr('height', toolTipBoundingBoxHeight);
            ephemeralTextLinesGroup.selectAll('*')
                .attr('x', toolTipX)
                .attr('y', toolTipY);
            const elementsSelection = toolTipGroup.selectAll('*');
            elementsSelection
                .transition()
                .duration(toolTipTransitionTime)
                .style("opacity", desiredOpacity);
        };

        attentionScatterPoints.selectAll('circle').data(attention_data)
            .remove();
        attentionScatterPoints.selectAll('circle').data(attention_data)
            .enter()
            .append('circle')
            .on('mouseover', function(datum) {
                const [mouseX, mouseY] = d3.mouse(this);
                updateToolTip(mouseX, mouseY+100, 1, datum, attentionToolTipFill);
            })
            .on('mouseout', datum => {
                updateToolTip(-svgWidth, -svgHeight, 0, datum, attentionToolTipFill);
            })
            .attr('cx', datum => xScale(getDatumParameterCount(datum)))
            .attr('cy', datum => yScale(getDatumLoss(datum)))
            .attr('r', scatterPointRadius)
            .attr('fill', attentionFill)
            .attr('fill-opacity', scatterPointFillOpacity);
        
        plainRNNScatterPoints.selectAll('circle').data(plain_rnn_data)
            .remove();
        plainRNNScatterPoints.selectAll('circle').data(plain_rnn_data)
            .enter()
            .append('circle')
            .on('mouseover', function(datum) {
                const [mouseX, mouseY] = d3.mouse(this);
                updateToolTip(mouseX, mouseY+100, 1, datum, plainRNNToolTipFill);
            })
            .on('mouseout', datum => {
                updateToolTip(-svgWidth, -svgHeight, 0, datum, plainRNNToolTipFill);
            })
            .attr('cy', datum => yScale(getDatumLoss(datum)))
            .attr('cx', datum => xScale(getDatumParameterCount(datum)))
            .attr('r', scatterPointRadius)
            .attr('fill', plainRNNFill)
            .attr('fill-opacity', scatterPointFillOpacity);

    };

    d3.json(data_location)
        .then(data => {
            let extract_data = datum => {
                return {
                    number_of_epochs: parseInt(datum.number_of_epochs),
                    batch_size: parseInt(datum.batch_size),
                    vocab_size: parseInt(datum.vocab_size),
                    pre_trained_embedding_specification: datum.pre_trained_embedding_specification,
                    encoding_hidden_size: parseInt(datum.encoding_hidden_size),
                    number_of_encoding_layers: parseInt(datum.number_of_encoding_layers),
                    attention_intermediate_size: parseInt(datum.attention_intermediate_size),
                    number_of_attention_heads: parseInt(datum.number_of_attention_heads),
                    dropout_probability: parseFloat(datum.dropout_probability),
                    test_loss: parseFloat(datum.test_loss),
                    test_accuracy: parseFloat(datum.test_accuracy),
                    number_of_parameters: parseInt(datum.number_of_parameters)
                };
            };
            let attention_data = data['attention'].map(extract_data);
            let plain_rnn_data = data['plain_rnn'].map(extract_data);
            const redraw = () => {
                render(attention_data, plain_rnn_data);
            };
            redraw();
            window.addEventListener('resize', redraw);
        }).catch(err => {
            console.error(err.message);
            return;
        });
};

scatterPlotMain();
